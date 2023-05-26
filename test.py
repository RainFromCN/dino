import unittest
import vit

import torch
import dino
import data
import config


class TestViT(unittest.TestCase):
    def test_vit_small(self):
        model = vit.vit_small(drop_path=0.1)
        x = torch.rand(32, 3, 96, 96)
        output = model(x)

        self.assertTrue(output.shape[0] == 32)
        self.assertTrue(output.shape[1] == model.embed_dim)


class TestDINO(unittest.TestCase):
    def test_dino_head(self):
        model = dino.DINOHead(input_dim=384, feat_dim=512)

        # 第一个测试用例
        x = torch.rand(32,3,384)
        output = model(x)
        self.assertTrue(output.shape == (32,3,512))

        # 第二个测试用例
        x = torch.rand(32,3,4,5,384)
        output = model(x)
        self.assertTrue(output.shape == (32,3,4,5,512))


    def test_dino_loss(self):
        feat_dim = 512
        model = dino.DINOLoss(feat_dim=feat_dim, momentum_center=0.9)

        # 第一个测试用例
        student_outs = [torch.rand(16,feat_dim) for _ in range(10)]
        teacher_outs = [torch.rand(16,feat_dim) for _ in range(2)]
        output = model(student_outs, teacher_outs, 1, 1)
        self.assertTrue(output.item() > 0)


    def test_feature_extractor(self):
        feat_dim = 512

        # 第一个测试用例
        model = dino.DINOFeatExtractor(feat_dim=feat_dim,
                                       drop_path=0.1, norm_last_layer=False)
        x = [torch.rand(2,3,48,48) for _ in range(2)]
        x += [torch.rand(2,3,64,64) for _ in range(3)]
        x += [torch.rand(2,3,32,32) for _ in range(4)]
        feats = model(x)
        self.assertTrue(len(feats) == 9)
        self.assertTrue(feats[0].shape == (2,feat_dim))

        # 第二个测试用例
        model = dino.DINOFeatExtractor(feat_dim=feat_dim,
                                       drop_path=0, norm_last_layer=True)
        x = [torch.rand(2,3,32,16) for _ in range(2)]
        x += [torch.rand(2,3,16,64) for _ in range(3)]
        x += [torch.rand(2,3,32,48) for _ in range(4)]
        feats = model(x)
        self.assertTrue(len(feats) == 9)
        self.assertTrue(feats[0].shape == (2,feat_dim))

    
    def test_dino(self):
        feat_dim = 512
        model = dino.DINO(feat_dim)

        # 第一个测试用例
        images = [torch.rand(2,3,64,64) for _ in range(2)]
        images += [torch.rand(2,3,32,32) for _ in range(4)]
        output = model(images, 1, 1)
        self.assertTrue(output.item() > 0)
        output.backward()




class TestData(unittest.TestCase):
    def test_dataloader(self):
        # 测试用例一
        dataset = data.get_dataset(is_train=True)
        data_loader = data.get_dataloader(dataset)
        for images, _ in data_loader:
            self.assertTrue(isinstance(images, list))
            self.assertTrue(len(images) == config.NUM_LOCAL_CROPS + 2)
            self.assertTrue(images[0].shape == (config.BATCH_SIZE, 3, config.GLOBAL_CROP_SIZE, config.GLOBAL_CROP_SIZE))
            self.assertTrue(images[1].shape == (config.BATCH_SIZE, 3, config.GLOBAL_CROP_SIZE, config.GLOBAL_CROP_SIZE))
            for i in range(2, len(images)):
                self.assertTrue(images[i].shape == (config.BATCH_SIZE, 3, config.LOCAL_CROP_SIZE, config.LOCAL_CROP_SIZE))
            break

        # 测试用例二
        dataset = data.get_dataset(is_train=False)
        data_loader = data.get_dataloader(dataset)
        for images, _ in data_loader:
            self.assertTrue(isinstance(images, list))
            self.assertTrue(len(images) == config.NUM_LOCAL_CROPS + 2)
            self.assertTrue(images[0].shape == (config.BATCH_SIZE, 3, config.GLOBAL_CROP_SIZE, config.GLOBAL_CROP_SIZE))
            self.assertTrue(images[1].shape == (config.BATCH_SIZE, 3, config.GLOBAL_CROP_SIZE, config.GLOBAL_CROP_SIZE))
            for i in range(2, len(images)):
                self.assertTrue(images[i].shape == (config.BATCH_SIZE, 3, config.LOCAL_CROP_SIZE, config.LOCAL_CROP_SIZE))
            break

if __name__ == '__main__':
    unittest.main()
