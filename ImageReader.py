class ImageReader:
    def __init__(self,imageFile, imageSize):
        self.imageFile = imageFile
        self.imageSize = int(imageSize)
        self.images = []
        self.answers = []

    def readImages(self):
        #check to make sure images have not yet been initialized
        if len(self.images) == 0:
            file = open(self.imageFile, "r+")
            inp = file.readlines()
            #cleans the lines
            inp = [line.strip("\n") for line in inp]
            if self.imageSize == 32:
                """32"""
                image = []
                for r in inp:
                    row = r.split(" ")
                    if len(row) == 2:
                        #digit answer
                        try:
                            if len(image) > 0:
                                self.images += [image]
                                image = []
                            digit = int(row[1])
                            self.answers += [digit]
                        except:
                            continue
                    elif len(row) == 1:
                            digits = list(row[0])
                            for digit in digits:
                                try:
                                    image += [int(digit)]
                                except:
                                    continue
            elif self.imageSize == 8:
                """8"""
                for r in inp:
                    row = r.split(",")
                    if len(row) > 0:
                        input = [int(digit) for digit in row]
                        answer = input[-1]
                        image = input[:-1]
                        self.images += [image]
                        self.answers += [answer]
            else:
                print("INVALID IMAGE SIZE")


    def getImages(self):
        return self.images, self.answers
