import pandas

from preprocessingutils import PreprocessingUtils


class Inference():
    def __init__(self,pclfile,categoriesfile,testfile):

        self.pclfile = pclfile
        self.categoriesfile = categoriesfile
        self.testfile = testfile

        self.preprocess = PreprocessingUtils(pclfile,categoriesfile,testfile)




def main():
    pass


if __name__ == "__main__":
    main()

