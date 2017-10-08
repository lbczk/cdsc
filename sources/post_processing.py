# The purpose of this class is to "compile" the predictions in the form of a np.array and turn them into a csv
# in the appropriate format.

# We need to specify in which way to aggregate the predictions for different pictures for a given object.
# Later we also want to combine predictions of different category levels and use a fallback if the predictions
# at level 3 are too bad.
import numpy as np
import pandas as pd


class PostProcessing(object):
    def __init__(self, predictions, num_imgs, product_id, idx2cat={}):
        self.number_of_images = predictions.shape[1]
        assert len(idx2cat.keys()) == self.number_of_images
        assert num_imgs.sum() == predictions.shape[0]
        self.predictions = predictions  # memory?
        self.num_imgs = num_imgs
        self.agreg_done = False
        self.product_id = product_id
        self.idx2cat = idx2cat

    def agreg(self, agreg_mode=0):
        self.agreg_done = True
        res = np.empty(self.num_imgs.shape[0])
        z = 0
        for i in xrange(self.num_imgs.shape[0]):
            aux = np.zeros(self.number_of_images)
            for j in xrange(int(self.num_imgs[i])):
                print("i, j --> {} {}".format(i, j))
                if agreg_mode == 0:
                    aux += self.predictions[z + j]
                else:
                    aux = np.max(aux, self.predictions[z + j])
            res[i] = np.argmax(aux)
            z += 1 + j

        print("Final z: {}".format(z))
        self.res = res
        return None

    def build_df(self):
        if not self.agreg_done:
            self.agreg(0)
        print("The first 5 predictions : {}".format(self.res[:10]))

        new_res = np.empty(self.res.shape[0])
        for k, v in self.idx2cat.iteritems():
            new_res[self.res == k] = v

        self.new_res = new_res.astype(int)
        self.res_df = pd.DataFrame({"_id": self.product_id, "category_id": self.new_res})
        return None

arti_preds = np.random.rand(80, 10)
arti_num_imgs = np.ones(60)
arti_num_imgs[:20] = 2
arti_product_id = range(60)
arti_idx2cat = {i: 1000 + i for i in range(10)}
pp = PostProcessing(arti_preds, arti_num_imgs, arti_product_id, idx2cat=arti_idx2cat)
pp.agreg(0)
