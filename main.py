from PIL import Image
import json
import utils

import torch
import torch.utils.data as data
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import tqdm

import logging
from argparse import Namespace

from functools import lru_cache

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

from transformers.tokenization_bert import BertTokenizer
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer


from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from typing import List, Dict
import os

import warnings
warnings.filterwarnings("ignore")

#Sees everything
pl.trainer.seed_everything(seed=42)


class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP) nj
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)
        bb_ot = h
        
        # construct positional encodings
        """        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)"""

        bs,_,H, W = h.shape
        pos = torch.cat([
        self.col_embed[:W].unsqueeze(0).unsqueeze(1).repeat(bs,H, 1, 1),
        self.row_embed[:H].unsqueeze(0).unsqueeze(2).repeat(bs,1, W, 1),
        ], dim=-1).flatten(1, 2)


        #print(self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1))
        # propagate through the transformer
        #shape changed to (W*H,bs,hidden_dim) for both pos and h
        h = self.transformer(pos.permute(1, 0, 2) + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1).repeat(1,bs,1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid(),
                'decoder_out':h,
                'res_out':bb_ot}

class VQA_DETR(LightningModule):
    def __init__(self,hparams,root,hidden_size=256, num_attention_heads = 8, num_hidden_layers = 6):
        super().__init__()

        self.hparams = hparams
        self.root = root
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
        self.bert_decoder_config = BertConfig(is_decoder = True,hidden_size=hidden_size, num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)
        #self.enc_dec_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config= self.bert_config, decoder_config= self.bert_config)
        #self.model = EncoderDecoderModel(config= self.enc_dec_config)
        self.bert_decoder = BertModel(config=self.bert_decoder_config)

        self.detr = DETRdemo(num_classes=91)
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
            map_location='cpu', check_hash=True)
        self.detr.load_state_dict(state_dict)
        del state_dict
        
        self.ans_to_index = self._mapping_ansto_index()

        self.classifier  = nn.Linear(hidden_size*2,len(self.ans_to_index))

        self.drop_out = nn.Dropout(p=0.2)
        self.log_softmax = nn.LogSoftmax().cuda()
        

    def forward(self,img, q_ids):
        
        img_ecs = self.detr(img)['decoder_out'].flatten(2)
        o1,_ = self.bert_decoder(input_ids = q_ids, encoder_hidden_states = img_ecs)

        mean_pool = torch.mean(o1,1)
        max_pool,_ = torch.max(o1,1)
        cat = torch.cat((mean_pool, max_pool),1)

        bo = self.drop_out(cat)
        output = self.classifier(bo)
        
        nll = -self.log_softmax(output)

        return {'logits':output,'nll':nll}


    def training_step(self, batch, batch_idx):
        im,q,a  = batch
        ids = q["ids"]

        outputs = self(im,ids)
        output_nll =outputs['nll']
        logits =  outputs['logits']

        loss = self.loss_fn(output_nll, a)
        f1 = self.metric_f1(logits, a)
        tensorboard_logs = {'train_loss': loss,'train_f1_score': f1}

        return {'loss': loss, 'log': tensorboard_logs,"progress_bar": {'train_loss': loss,'train_f1':f1}}

    def validation_step(self, batch, batch_idx):
        im,q,a  = batch
        ids = q["ids"]

        outputs = self(im,ids)
        output_nll =outputs['nll']
        logits =  outputs['logits']

        loss = self.loss_fn(output_nll, a)
        f1 = self.metric_f1(logits, a)
        tensorboard_logs = {'val_loss': loss,'val_f1_score': f1}

        return {'val_loss': loss, 'log': tensorboard_logs,"progress_bar": {'val_loss': loss,'val_f1':f1}}


    def validation_end(self, outputs: List[dict]):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        f1 = torch.stack([x['progress_bar']['val_f1'] for x in outputs]).mean()
        return {"val_loss": loss,"val_f1":f1}

    def loss_fn(self, nll, targets):

        return (nll * targets / 10).sum(dim=1).mean()#nn.CrossEntropyLoss()(outputs, targets)

    
    
    def metric_f1(self, preds, y):

        _, max_preds = preds.max(dim = -1) # get the index of the max 
        _, y = y.max(dim= -1)
        shape = max_preds.shape[0]
        f1=f1_score(y.cpu().view(shape).numpy(),max_preds.cpu().view(shape).numpy(),average='macro')
        f1  = torch.tensor(f1, dtype  = torch.float32)
        return f1

    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.accumulate_grad_batches * self.hparams.epochs


    def configure_optimizers(self):

        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        optimizer = AdamW(optimizer_parameters, lr=self.hparams.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.total_steps(),
        )

        return [optimizer],  [{"scheduler": scheduler, "interval": "step"}]

    def _mapping_ansto_index(self):
        # compile a list of all the answers
        entries = []
        for name in ['train', 'val']:
            entries += utils._load_dataset(self.root,name)
        all_answers  = set()
        for a in entries:
            all_answers.update(a['answer'])
        all_answers=list(all_answers)

        answer_to_index = dict()
        for i,answer in enumerate(all_answers):
            answer_to_index[answer]=i

        return answer_to_index

    def prepare_data(self):
        
        self.val_dataset  = VQA(root=self.root, answer_to_index=self.ans_to_index,split= 'val', tokenizer=self.tokenizer, max_len=15 )
        self.train_dataset  = VQA(root=self.root, answer_to_index=self.ans_to_index,split= 'train', tokenizer=self.tokenizer, max_len=15 )

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size = self.hparams.batch_size,num_workers=self.hparams.num_workers, shuffle= True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size = self.hparams.val_batch_size,num_workers=self.hparams.num_workers, shuffle= False)
        return loader


class VQA(data.Dataset):
    """ VQA dataset, open-ended """
    def __init__(self, root, answer_to_index, tokenizer ,split = 'train', max_len = 20):
        super(VQA, self).__init__()


        self.root = root
        self.answer_to_index = answer_to_index
        self.split = split
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.entries = utils._load_dataset( self.root, self.split)

         # standard PyTorch mean-std input image normalization
        self.transform = T.Compose([
            T.Resize(size=(800,800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.id_to_image_fname = self._find_iamges()


    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        
        inputs = self.tokenizer.encode_plus(
            question,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_len - len(ids)
        ids += ([0]*padding_length)
        mask += ([0]*padding_length)
        token_type_ids += ([0]*padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            
        }

    def _encode_answer_bert(self, answers):
        pass


    def _encode_answers(self, answers):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        answer_vec = torch.zeros(len(self.answer_to_index),dtype=torch.long)
        for answer in answers:
            index = self.answer_to_index.get(answer)
            if index is not None:
                answer_vec[index] += 1
        return answer_vec

   
    def _find_iamges(self):
        id_to_filename = {}
        imgs_folder = os.path.join(self.root,'%s2014'%self.split)
        for filename in os.listdir(imgs_folder):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = os.path.join(imgs_folder,filename)
        return id_to_filename


    def _load_image(self, image_id):
        """ Load an image """

        img_path = self.id_to_image_fname[image_id]
        img  = Image.open(img_path)
        img = np.asarray(img)
        
        if len(img.shape)==2:
            img=np.expand_dims(img, axis=-1)
            img = np.repeat(img,3, axis = -1)   
        return img

    def __getitem__(self, item):
       
        entry  = self.entries[item]
        image_id = entry['image_id']
        try:
            img = self._load_image(image_id)
        except:
            entry = self.entries[0]
            image_id = entry['image_id']
            img = self._load_image(image_id)

        q = entry['question']
        a = self._encode_answers(entry['answer'])
        img = Image.fromarray(img)
        img = self.transform(img)
        #question_id = entry['question_id']
        q= self._encode_question(q)

        return img, q, a

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def collate_fn(batch):
        """The collat_fn method to be used by the
        PyTorch data loader.
        """
        # Unzip the batch
        imgs,qs, answers = list(zip(*batch))

        # concatenate the vectors
        imgs = torch.stack(imgs)
        
        #concatenate the labels
        q = torch.stack(qs)
        a = torch.stack(answers)
        
        return imgs, q, a


if __name__=="__main__":
    hparams = Namespace(
        batch_size=3,
        val_batch_size=3,
        num_warmup_steps=100,
        epochs=20,
        lr=3e-5,
        accumulate_grad_batches=1,
        num_workers = 1
    )

    vqa_detr = VQA_DETR(hparams=hparams,root="/home/ubuntu/vqa/vqa_data")
    trainer = Trainer(gpus=4, max_epochs=20,train_percent_check=.2,log_gpu_memory=True,weights_summary=None)
    #trainer = Trainer(fast_dev_run=True)
    trainer.fit(vqa_detr)