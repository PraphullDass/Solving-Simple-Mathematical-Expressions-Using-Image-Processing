# importing all required libraries
import numpy as np
import cv2
import tensorflow as tf
import tkinter
from PIL import Image, ImageTk

# function to resize all images to match size required by our CNN model
def resize_images(parts):
    parts_resized = []
    for part in parts:
        parts_resized.append(cv2.resize(part, (60, 60)))
    return parts_resized

# recursive function to know about the largest connected component
def get_component(i, j, visited, i_min, i_max, j_min, j_max):
    if(not visited[i][j]):
        visited[i][j] = True
        i_min = min(i_min, i)
        i_max = max(i_max, i)
        j_min = min(j_min, j)
        j_max = max(j_max, j)
        
        if(not visited[i-1][j-1]):
            x1, x2, y1, y2 = get_component(i-1, j-1, visited, i_min, i_max, j_min, j_max)
            i_min = min(i_min, x1)
            i_max = max(i_max, x2)
            j_min = min(j_min, y1)
            j_max = max(j_max, y2)
        if(not visited[i-1][j]):
            x1, x2, y1, y2 = get_component(i-1, j, visited, i_min, i_max, j_min, j_max)
            i_min = min(i_min, x1)
            i_max = max(i_max, x2)
            j_min = min(j_min, y1)
            j_max = max(j_max, y2)
        if(not visited[i-1][j+1]):
            x1, x2, y1, y2 = get_component(i-1, j+1, visited, i_min, i_max, j_min, j_max)
            i_min = min(i_min, x1)
            i_max = max(i_max, x2)
            j_min = min(j_min, y1)
            j_max = max(j_max, y2)
        if(not visited[i][j-1]):
            x1, x2, y1, y2 = get_component(i, j-1, visited, i_min, i_max, j_min, j_max)
            i_min = min(i_min, x1)
            i_max = max(i_max, x2)
            j_min = min(j_min, y1)
            j_max = max(j_max, y2)
        if(not visited[i][j+1]):
            x1, x2, y1, y2 = get_component(i, j+1, visited, i_min, i_max, j_min, j_max)
            i_min = min(i_min, x1)
            i_max = max(i_max, x2)
            j_min = min(j_min, y1)
            j_max = max(j_max, y2)
        if(not visited[i+1][j-1]):
            x1, x2, y1, y2 = get_component(i+1, j-1, visited, i_min, i_max, j_min, j_max)
            i_min = min(i_min, x1)
            i_max = max(i_max, x2)
            j_min = min(j_min, y1)
            j_max = max(j_max, y2)
        if(not visited[i+1][j]):
            x1, x2, y1, y2 = get_component(i+1, j, visited, i_min, i_max, j_min, j_max)
            i_min = min(i_min, x1)
            i_max = max(i_max, x2)
            j_min = min(j_min, y1)
            j_max = max(j_max, y2)
        if(not visited[i+1][j+1]):
            x1, x2, y1, y2 = get_component(i+1, j+1, visited, i_min, i_max, j_min, j_max)
            i_min = min(i_min, x1)
            i_max = max(i_max, x2)
            j_min = min(j_min, y1)
            j_max = max(j_max, y2) 
        
    return i_min, i_max, j_min, j_max

# function to divide image in multible parts and return all parts as a list
def get_parts(img):
    m, n = img.shape
    m_half = m//2
    visited = np.zeros(shape = (m, n), dtype = bool)
    visited[img == 0] = True

    parts = []

    j = 0

    while (j < n):
        for i in range(m):
            if(not visited[i][j]):
                i1, i2, j1, j2 = get_component(i,j,visited, i, i, j, j)
                if(j2 > j):
                    j = j2
                i_diff = i2-i1
                j_diff = j2-j1
                width = max(i_diff, j_diff) + 10
                temp = img[i1:i2,j1:j2].copy()
                temp2 = np.zeros(shape = (width, width), dtype = np.uint8)
                i_start = width//2-i_diff//2
                i_end = i_start + i_diff
                j_start = width//2-j_diff//2
                j_end = j_start + j_diff
                temp2[i_start:i_end,j_start:j_end] = temp.copy()
                parts.append(temp2)
        j += 1

    return parts

# function to pad the cropped image
def get_pad_image(img):
    m, n = img.shape
    pad_image = np.zeros(shape = (m + 10, n + 10), dtype = np.uint8)
    pad_image[5 : 5 + m, 5 : 5 + n] = img.copy()
    return pad_image

# function to crop the image and obtain the import part 
def get_cropped_image(img):
    m, n = img.shape
    i1, i2, j1, j2 = 0, m-1, 0, n-1
    visited = np.zeros(shape = (m, n), dtype = bool)
    visited[img == 0] = True
    
    for i in range(0, m, 5):
        if(img[i,:].sum() != 0 and img[i+5,:].sum() != 0):
            i1 = i-5
            break
    for i in range(i1 + 5, m, 5):
        if(img[i,:].sum() == 0):
            i2 = i + 5
            break
    
    img = img[i1-1:i2+1,:]
    
    for j in range(0, n, 5):
        if(img[:,j].sum() != 0 and img[:,j+5].sum() != 0):
            j1 = j - 5
            break
    for j in range(n-1, -1, -5):
        if(img[:,j].sum() != 0 and img[:,j-5].sum() != 0):
            j2 = j + 5
            break
    
    return img[:,j1-1:j2+1]

# function to perform thresholding which will make our required text statnd out
def threshold(img, k):
    img[img >= k] = 255
    img[img < k] = 0
    return img

# function to get the best k for thresholding
# we are using histogram of the image to know about this value
def get_thresh_k(img):
    counts = []
    for i in range(0, 256):
        counts.append((img == i).sum())

    k = 255
    for i in range(254, 0, -2):
        if((sum(counts[:i])/(sum(counts[i:])+1)) > 100):
            k = i
    return k

# function t remove light shadows
def remove_shadow(img):
    kernel = np.ones(shape = (5,5), dtype = np.uint8)
    dilated_image = cv2.dilate(img.copy(), kernel)
    bg_image = cv2.medianBlur(dilated_image.copy(), 5)
    res = cv2.absdiff(img.copy(), bg_image.copy())
    return res

# function of obtain image
def get_image(image_name):
    image = cv2.resize(cv2.imread(image_name, 0), (576, 324))
    return image

# function to scale image for CNN model
def remodel_parts(parts):
    parts_remodeled = []
    for part in parts:
        parts_remodeled.append(part.astype(float).reshape(60,60,1) / 255)
    return parts_remodeled

# function to get labels for the images after prediction from the CNN model
def get_labels(y_predicted):
    mapping = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"(", 11:")", 12:"+", 13:"-", 14:"*", 15:"/"}
    y = ""
    for y_pred in y_predicted:
        y += mapping[y_pred.argmax()]
    return y

# function to display all images as a art gallery using tkinter
def mytkinter(images, names):
    size = len(images)
    root = tkinter.Tk()
    root.title("DIP Project")
    
    image_list = []
    for img in images:
        img = np.array(img)
        im = Image.frombytes('L', (img.shape[1],img.shape[0]), img.astype('b').tostring())
        photo = ImageTk.PhotoImage(image = Image.fromarray(img.astype("uint8")))
        image_list.append(photo)
    
    img = np.array(images[0])
    im = Image.frombytes('L', (img.shape[1],img.shape[0]), img.astype('b').tostring())
    photo = ImageTk.PhotoImage(image = Image.fromarray(img.astype("uint8")))
    
    global my_label
    global button_back
    global button_name
    global button_forward
    
    my_label = tkinter.Label(root, image = photo)
    my_label.grid(row=0, column=0, columnspan = 3)
    
    # function to mode slide show forward
    def forward(image_number):
        global my_label
        global button_forward
        global button_name
        global button_back
        
        my_label.grid_forget()
        button_name.grid_forget()
        my_label = tkinter.Label(image = image_list[image_number - 1])
        button_forward = tkinter.Button(root, text = ">>", command = lambda : forward(image_number + 1))
        button_name = tkinter.Button(root, text = names[image_number - 1], state = tkinter.DISABLED)
        button_back = tkinter.Button(root, text = "<<", command = lambda : back(image_number - 1))
        
        if(image_number == size):
            button_forward = tkinter.Button(root, text = ">>", state = tkinter.DISABLED)
        
        my_label.grid(row=0, column=0, columnspan = 3)
        button_back.grid(row=1, column=0)
        button_name.grid(row=1, column=1)
        button_forward.grid(row=1, column=2)
        
        return
    
    # function to mode slide show backward
    def back(image_number):
        global my_label
        global button_forward
        global button_name
        global button_back
        
        my_label.grid_forget()
        button_name.grid_forget()
        my_label = tkinter.Label(image = image_list[image_number - 1])
        button_forward = tkinter.Button(root, text = ">>", command = lambda : forward(image_number + 1))
        button_name = tkinter.Button(root, text = names[image_number - 1], state = tkinter.DISABLED)
        button_back = tkinter.Button(root, text = "<<", command = lambda : back(image_number - 1))
        
        if(image_number == 1):
            button_back = tkinter.Button(root, text = "<<", state = tkinter.DISABLED)
        
        my_label.grid(row=0, column=0, columnspan = 3)
        button_back.grid(row=1, column=0)
        button_name.grid(row=1, column=1)
        button_forward.grid(row=1, column=2)
        
        return
    
    button_back = tkinter.Button(root, text = "<<", command = back, state = tkinter.DISABLED)
    button_name = tkinter.Button(root, text = "Original Image", state = tkinter.DISABLED)
    button_forward = tkinter.Button(root, text = ">>", command = lambda : forward(2))
    
    button_back.grid(row=1, column=0)
    button_name.grid(row=1, column=1)
    button_forward.grid(row=1, column=2)
    
    root.mainloop()
    return

# out main function
def run(filename, modelname):
    image_org = get_image(filename)
    image = 255 - image_org.copy()
    k = get_thresh_k(image.copy())
    thresh_image = threshold(image.copy(), k)
    cropped_image = get_cropped_image(thresh_image.copy())
    pad_image = get_pad_image(cropped_image.copy())
    parts = get_parts(pad_image.copy())
    parts_resized = resize_images(parts.copy())
    #cv2.imshow("original", image)
    #cv2.imshow("thresh", thresh_image)
    #cv2.imshow("cropped", cropped_image)
    #cv2.imshow("pad_image", pad_image)
    #cv2.waitKey(0)
    parts_remodeled = remodel_parts(parts_resized.copy())
    model = tf.keras.models.load_model(modelname)
    y_predicted = model.predict(np.array(parts_remodeled.copy()))
    y = get_labels(y_predicted.copy())
    print("Your expression is : " + str(y))
    print("Your answer is : " + str(eval(y)))
    temp = [image_org, image, thresh_image, pad_image]
    temp += parts_resized
    names = ["Original Image", "Negative Image", "Thresholded Imgae", "Cropped Image"]
    for x in parts_resized:
        names.append("Parts")
    mytkinter(temp, names)
    return

# calling main function
run("PRAPHULL_DASS_2018071_DEMO2.jpg", "PRAPHULL_DASS_2018071_CNN_MODEL")
