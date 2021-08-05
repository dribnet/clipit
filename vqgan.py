# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

from DrawingInterface import DrawingInterface

import sys
sys.path.append('taming-transformers')
import os.path
import torch

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan

vqgan_config_table = {
    "imagenet_f16_1024": 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.yaml',
    "imagenet_f16_16384": 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.yaml',
    "openimages_f16_8192": 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
    "coco": 'https://dl.nmkd.de/ai/clip/coco/coco.yaml',
    "faceshq": 'https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT',
    "wikiart_1024": 'http://mirror.io.community/blob/vqgan/wikiart.yaml',
    "wikiart_16384": 'http://mirror.io.community/blob/vqgan/wikiart_16384.yaml',
    "sflckr": 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1',
}
vqgan_checkpoint_table = {
    "imagenet_f16_1024": 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.ckpt',
    "imagenet_f16_16384": 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.ckpt',
    "openimages_f16_8192": 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1',
    "coco": 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt',
    "faceshq": 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt',
    "wikiart_1024": 'http://mirror.io.community/blob/vqgan/wikiart.ckpt',
    "wikiart_16384": 'http://mirror.io.community/blob/vqgan/wikiart_16384.ckpt',
    "sflckr": 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1'
}

class VqganDrawer(DrawingInterface):
    def load_model(self, config_path, checkpoint_path, device):
        gumbel = False
        config = OmegaConf.load(config_path)
        if config.model.target == 'taming.models.vqgan.VQModel':
            model = vqgan.VQModel(**config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(checkpoint_path)
        elif config.model.target == 'taming.models.vqgan.GumbelVQ':
            model = vqgan.GumbelVQ(**config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(checkpoint_path)
            gumbel = True
        elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            parent_model.init_from_ckpt(checkpoint_path)
            model = parent_model.first_stage_model
        else:
            raise ValueError(f'unknown model type: {config.model.target}')
        del model.loss

        # model, gumbel = load_vqgan_model(vqgan_config, vqgan_checkpoint)
        self.model = model.to(device)
        self.gumbel = gumbel
        self.device = device

        if gumbel:
            self.e_dim = 256
            self.n_toks = model.quantize.n_embed
            self.z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
            self.z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
        else:
            self.e_dim = model.quantize.e_dim
            self.n_toks = model.quantize.n_e
            self.z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
            self.z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    def rand_init(self, toksX, toksY):
        # legacy init
        one_hot = F.one_hot(torch.randint(self.n_toks, [toksY * toksX], device=self.device), n_toks).float()
        if self.gumbel:
            self.z = one_hot @ self.model.quantize.embed.weight
        else:
            self.z = one_hot @ self.model.quantize.embedding.weight

        self.z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        self.z.requires_grad_(True)

    def init_from_tensor(self, init_tensor):
        self.z, *_ = self.model.encode(init_tensor)        
        self.z.requires_grad_(True)

    def clip_z(self):
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

    def get_z(self):
        return self.z

        # return model, gumbel

### EXTERNAL INTERFACE
### load_vqgan_model

if __name__ == '__main__':
    main()
