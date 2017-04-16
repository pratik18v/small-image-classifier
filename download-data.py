import urllib
import os
from progressbar import ProgressBar
import glob
from PIL import Image

def download_image(url, label):
    
    #Images saved in this directory
    directory = 'images/'
    
    #Get image links
    try:
        f = urllib.urlopen(url)
    except:
        print 'Invalid URL {}'.format(url)
        return
    
    link = f.read().split('\n')
    
    print 'Downloading images ...'
    count = 0       #Keep a count of invalid image links
    pbar = ProgressBar()
    for i in pbar(range(len(link)-1)):
        try:
            #Check if image already exists
            if os.path.isfile(directory + label + "-" + str(i) + ".jpg") == True:
                continue
            else:
                urllib.urlretrieve(link[i], directory + label + "-" + str(i) + ".jpg")
        except:
            print 'Not found: {}'.format(link[i])
            count += 1
            
    print '{} images not found'.format(count)

def remove_invalids(folder):

    imgnames = []
    for fname in glob.glob(folder+'*.jpg'):
        imgnames.append(fname)
     
    pbar = ProgressBar()
    print 'Deleting invalid images ...'
    for i in pbar(range(len(imgnames))):
        try:
            im = Image.open(imgnames[i])
            pixels = im.getdata()
            
            black_thresh = (0,0,0,0)
            white_thresh = 255
            nblack = 0
            for pixel in pixels:
                if pixel == black_thresh or pixel == white_thresh:
                    nblack += 1
            n = len(pixels)
            
            if (nblack / float(n)) > 0.8:
                os.remove(imgnames[i])
        except:
            os.remove(imgnames[i])
            
        
if __name__ == '__main__':

    #URLs provided by the coding challenge
    url = ['http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02084071', \
    'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02121808', \
    'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00007846']

    #Download images
    for i in range(len(url)):
        download_image(url[i], str(i+1))
        
    #Remove invalid images (pass image directory)
    remove_invalids('images/')