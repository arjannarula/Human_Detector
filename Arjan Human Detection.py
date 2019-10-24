#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
from math import sqrt

from math import degrees





# In[55]:



#  Prewitt Operator  used for the Prewitt Operation and wil return the GradientX and GradientY. 
def prewitt_operator(new_image_array):
    perwittoperatorsx=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    perwittoperatorsxx,perwittoperatorsxy=perwittoperatorsx.shape
    perwittoperatorsy=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    perwittoperatorsxx1,perwittoperatorsxy1=perwittoperatorsy.shape
    prewitt_operator_sx_median=perwittoperatorsxx//2
    prewitt_operator_sx1_median=perwittoperatorsxy//2
    prewitt_operator_sy_median=perwittoperatorsxx1//2
    prewitt_operator_sy1_median=perwittoperatorsxy1//2
    x,y=new_image_array.shape
    
    Gradientx=np.zeros((x,y),dtype=float)
    Gradienty=np.zeros((x,y),dtype=float)

    
    

    for i in range(x):
        for j in range(y):
            a=i+prewitt_operator_sx_median
            b=i-prewitt_operator_sy_median
            if((a) in range(x)):
                if((b) in range(x)):
                    c=j+prewitt_operator_sx1_median
                    d=j-prewitt_operator_sy1_median
                    if((c) in range(y)):
                        if((d) in range(y)):
                            start_row=b
                            sum1=0
                            sum2=0
                            for m in range(perwittoperatorsxx):
                                start_col=d
                                for l in range(perwittoperatorsxy):
                                    sum1=(sum1+(perwittoperatorsx[m][l]*new_image_array[start_row][start_col]))
                                    sum2=(sum2+(perwittoperatorsy[m][l]*new_image_array[start_row][start_col]))
                                    start_col=start_col+1
                                start_row=start_row+1
                            
                            Gradientx[i][j]=sum1
                            Gradienty[i][j]=sum2
    
    
    
                            
    plt.imshow(Gradientx)
    plt.show()     
    plt.imshow(Gradienty)
    plt.show()
    return Gradientx,Gradienty
    
    


# In[56]:


#Used for the Purpose of Finding Magnitude Array and Angle Array. It will return Magnitude and Angle Array/. 
def gradient_magnitude(Gradientx,Gradienty):
    x,y=Gradientx.shape
    Magnitudearray=np.zeros((x,y),dtype=float)
    anglearray1=np.zeros((x,y),dtype=float)
# Calculating Magnitude Array from the Grdient X and Gradient Y.    
    for a in range(x):
        for b in range(y):
            Magnitudearray[a][b]=((Gradientx[a][b]*Gradientx[a][b])+(Gradienty[a][b]*Gradienty[a][b]))
            Magnitudearray[a][b]=sqrt(Magnitudearray[a][b])
            Magnitudearray[a][b]=np.round(Magnitudearray[a][b]/1.4142)

            
            
#  Computing Angle Array from the Gradientx and GradientY            
    anglearray=np.arctan2(Gradienty,Gradientx)
    x,y=Gradientx.shape
    
# Noamalizing anglke      
    for e in range(x):
        for l in range(y):
            anglearray[e][l]=degrees(anglearray[e][l])
            anglearray[e][l]=np.mod((anglearray[e][l]+360),360)
            if((anglearray[e][l]>=170) and (anglearray[e][l]<350)):
                anglearray1[e][l]=anglearray[e][l]-180
            elif(anglearray[e][l]>=350):
                anglearray1[e][l]=anglearray[e][l]-360
            else:
                anglearray1[e][l]=anglearray[e][l]

    
    

    plt.imshow(Magnitudearray)
    plt.show()
    return Magnitudearray,anglearray1


# In[57]:


def histogram_of_gradient(Magnitudearray,anglearray):
    descriptor_list=[]
    x,y=Magnitudearray.shape

    list_of_histogram=[]
    i=0
    j=0
    while(i+16<=x):
            j=0
            while(j+16<=y):
                    sixteen_block=[]
                    cell_i=i
                    cell_j=j
                    count=0
                    l=0
#create list for the 16 cells means one block.                     
                    for l in range(4):
# Gives the eight cell                         
                        eight_cell=[0,0,0,0,0,0,0,0,0]
                        
                        k=0
                        p=0

                        for k in range(cell_i,cell_i+8):

                            for p in range(cell_j,cell_j+8):

                                if((anglearray[k][p]>=-10) and (anglearray[k][p]<0)):
                                    ratio1=(20-(np.abs(0+anglearray[k][p]))/20)
                                    ratio2=(20-(np.abs(20+anglearray[k][p]))/20)
                                    eight_cell[0]=eight_cell[0]+(Magnitudearray[k][p]*ratio1)
                                    eight_cell[8]=eight_cell[0]+(Magnitudearray[k][p]*ratio2)
                                elif((anglearray[k][p]>=0)and (anglearray[k][p]<=20)):
                                    if(anglearray[k][p]==0):
                                        eight_cell[0]=eight_cell[0]+Magnitudearray[k][p]
                                    elif(anglearray[k][p]==20):
                                        eight_cell[1]=eight_cell[1]+Magnitudearray[k][p]
                                    else:
                                        ratio1=(20-(np.abs(anglearray[k][p]-0))/20)
                                        ratio2=(20-(np.abs(anglearray[k][p]-20))/20)
                                        eight_cell[0]=eight_cell[0]+(Magnitudearray[k][p]*ratio1)
                                        eight_cell[1]=eight_cell[1]+(Magnitudearray[k][p]*ratio2)
                                elif((anglearray[k][p]>20)and (anglearray[k][p]<=40)):
                                    if(anglearray[k][p]==40):
                                        eight_cell[2]=eight_cell[2]+Magnitudearray[k][p]
                                    else:
                                        ratio1=(20-(np.abs(anglearray[k][p]-20))/20)
                                        ratio2=(20-(np.abs(anglearray[k][p]-40))/20)
                                        eight_cell[1]=eight_cell[1]+(Magnitudearray[k][p]*ratio1)
                                        eight_cell[2]=eight_cell[2]+(Magnitudearray[k][p]*ratio2)
                                elif((anglearray[k][p]>40)and (anglearray[k][p]<=60)):
                                    if(anglearray[k][p]==60):
                                        eight_cell[3]=eight_cell[3]+Magnitudearray[k][p]
                                    else:
                                        ratio1=(20-(np.abs(anglearray[k][p]-40))/20)
                                        ratio2=(20-(np.abs(anglearray[k][p]-60))/20)
                                        eight_cell[2]=eight_cell[2]+(Magnitudearray[k][p]*ratio1)
                                        eight_cell[3]=eight_cell[3]+(Magnitudearray[k][p]*ratio2)
                                elif((anglearray[k][p]>60) and (anglearray[k][p]<=80)):
                                    if(anglearray[k][p]==80):
                                        eight_cell[4]=eight_cell[4]+Magnitudearray[k][p]
                                    else:
                                        ratio1=(20-(np.abs(anglearray[k][p]-60))/20)
                                        ratio2=(20-(np.abs(anglearray[k][p]-80))/20)
                                        eight_cell[3]=eight_cell[3]+(Magnitudearray[k][p]*ratio1)
                                        eight_cell[4]=eight_cell[4]+(Magnitudearray[k][p]*ratio2)
                                elif((anglearray[k][p]>80) and (anglearray[k][p]<=100)):
                                    if(anglearray[k][p]==100):
                                        eight_cell[5]=eight_cell[5]+(Magnitudearray[k][p])
                                    else:
                                        ratio1=(20-(np.abs(anglearray[k][p]-80))/20)
                                        ratio2=(20-(np.abs(anglearray[k][p]-100))/20)
                                        eight_cell[4]=eight_cell[4]+(Magnitudearray[k][p]*ratio1)
                                        eight_cell[5]=eight_cell[5]+(Magnitudearray[k][p]*ratio2)
                                elif((anglearray[k][p]>100) and (anglearray[k][p]<=120)):
                                    if(anglearray[k][p]==120):
                                        eight_cell[6]=eight_cell[6]+Magnitudearray[k][p]
                                    else:
                                        ratio1=(20-(np.abs(anglearray[k][p]-100))/20)
                                        ratio2=(20-(np.abs(anglearray[k][p]-120))/20)
                                        eight_cell[5]=eight_cell[5]+(Magnitudearray[k][p]*ratio1)
                                        eight_cell[6]=eight_cell[6]+(Magnitudearray[k][p]*ratio2)
                                elif((anglearray[k][p]>120) and (anglearray[k][p]<=140)):
                                    if(anglearray[k][p]==140):
                                        eight_cell[7]=eight_cell[7]+Magnitudearray[k][p]
                                    else:
                                        ratio1=(20-(np.abs(anglearray[k][p]-120))/20)
                                        ratio2=(20-(np.abs(anglearray[k][p]-140))/20)
                                        eight_cell[6]=eight_cell[6]+(Magnitudearray[k][p]*ratio1)
                                        eight_cell[7]=eight_cell[7]+(Magnitudearray[k][p]*ratio2)
                                elif((anglearray[k][p]>140) and (anglearray[k][p]<=160)):
                                    if(anglearray[k][p]==160):
                                        eight_cell[8]=eight_cell[8]+Magnitudearray[k][p]
                                    else:
                                        ratio1=(20-(np.abs(anglearray[k][p]-140))/20)
                                        ratio2=(20-(np.abs(anglearray[k][p]-160))/20)
                                        eight_cell[7]=eight_cell[7]+(Magnitudearray[k][p]*ratio1)
                                        eight_cell[8]=eight_cell[8]+(Magnitudearray[k][p]*ratio2)
                                elif((anglearray[k][p]>160) and (anglearray[k][p]<170)):
                                    ratio1=(20-(np.abs(anglearray[k][p]-180))/20)
                                    ratio2=(20-(np.abs(anglearray[k][p]-160))/20)
                                    eight_cell[0]=eight_cell[0]+(Magnitudearray[k][p]*ratio1)
                                    eight_cell[8]=eight_cell[8]+(Magnitudearray[k][p]*ratio2)
                                
                                     
                                        
                                            
                                         
                                        
                    
                        if(count==0):
                            cell_j=cell_j+8
                            count=count+1
                        elif(count==1):
                            cell_i=cell_i+8
                            cell_j=j
                            count=count+1
                        elif(count==2):
                            cell_j=cell_j+8
                            count=count+1
                        sixteen_block.append(eight_cell)

                    
#creating a vector of thirty six and one.Normalizing the vector.  
                    thirty_six_vector=[]
                    yu=0
                    sum1=0
                    for yu in range(len(sixteen_block)):
                        c=sixteen_block[yu]
                        for hi in range(len(c)):
                            thirty_six_vector.append(c[hi])
                            sum1=sum1+(c[hi]*c[hi])
                    normalized_vector=[]
                    
                    square_root=np.sqrt(sum1)
                    if(square_root!=0):
                        for nb in range(len(thirty_six_vector)):
                            normalized_vector.append(thirty_six_vector[nb]/square_root)
# creating a descriptor list of length 7524.                    

                    
                        for kj in range(len(normalized_vector)):
                            descriptor_list.append(normalized_vector[kj])
                    else:
                        for nb in range(len(thirty_six_vector)):
                            normalized_vector.append(thirty_six_vector[nb])
                        for kj in range(len(normalized_vector)):
                            descriptor_list.append(normalized_vector[kj])
                        
            

                    j=j+8
            
            i=i+8
                    
                        
                                                 

    return descriptor_list

                                    
                    
            
        

            
        
               


# In[58]:


# Do prewitt operation,Gradient magnitude output and return a list of descriptor of length 7524.
def images_descriptor(image,imagename):
    

    b,g,r=cv2.split(image)
    new_image_array=np.round(0.299*r+0.587*g+0.114*b)
    x,y=new_image_array.shape
    Gradientx,Gradienty=prewitt_operator(new_image_array)
    Magnitudearray,anglearray1=gradient_magnitude(Gradientx,Gradienty)
    import imageio as i
    i.imwrite("Magnitude_ %s.bmp"%imagename,Magnitudearray)
    descriptor_list=histogram_of_gradient(Magnitudearray,anglearray1)
    return descriptor_list
    
    





# relu function
def relu(x):
    #print('hoho')
    n,m=x.shape
    #print(n,m)
    y=np.zeros((n,m),dtype=float)
    #print(x[0][27])
    for i in range(n):
        for j in range(m):
            if(x[i][j]>0):
                 y[i][j]=x[i][j]
        else:
                 y[i][j]=0
    #print(y[0][27])
    return y
# signoid function
def sigmoid(data):
   # print('hello')
    #print('-----------------',data)
    yy=1/(1+np.exp(-data))
    #print('-----------------------------y----------------',yy)
    return yy


# In[59]:


import math
# Neural network 
def neural(Descriptor_Array,Output_Array,number_of_hidden_nodes,eta,nodes_max):

    row_inp,col_inp=Descriptor_Array.shape
    no_example=row_inp
    n_input=col_inp
    row_out,col_out=Output_Array.shape
    n_output=col_out
    
# creating random array for the weights    
    node1=np.random.randn(n_input,number_of_hidden_nodes)
    node1=np.multiply(node1,math.sqrt(2/int(n_input+number_of_hidden_nodes)))
    
    node10=np.random.randn(number_of_hidden_nodes)
    node10=np.multiply(node10,math.sqrt(2/int(number_of_hidden_nodes)))
    
    node2=np.random.randn(number_of_hidden_nodes,n_output)
    node2=np.multiply(node2,math.sqrt(1/int(number_of_hidden_nodes+n_output)))
    
    node20=np.random.randn(n_output)
    node20=np.multiply(node20,math.sqrt(1/int(n_output)))
    

    for n in range(nodes_max):
        
        square_error= np.zeros((1,n_output))
# forward prpogation            
        for k in range(no_example):
            x=Descriptor_Array[k,:].reshape([1,-1])
            z=relu((x.dot(node1)+node10))

            y=sigmoid(z.dot(node2)+node20)
            
            
            err=(Output_Array[k,0]-y)
            
            square_error+= 0.5*np.square(err)
# backward propogation                       
            dell_output=(-1*err)*(1-y)*y
            dell_node2=z.T.dot(dell_output)
            dell_node20=np.sum(dell_output,axis=0)
            
            
            zz=np.zeros_like(z)
            for xyz in range(number_of_hidden_nodes):
            
                if(z[0][xyz]>0):
                    zz[0][xyz]=1
                else:
                    zz[0][xyz]=0
                       
            hidden_de_hidden= dell_output.dot(node2.T)*zz
            hidden_de_node1=x.T.dot(hidden_de_hidden)
            hidden_de_node10=np.sum(hidden_de_hidden,axis=0)
            
            node2-= eta*dell_node2
            node20-= eta*dell_node20
            node1-= eta*hidden_de_node1
            node10-= eta*hidden_de_node10
            
                       

    return node1,node10,node2,node20
   
# This is used as a function for predciting the output of the test images whether it detects correctly or not.It will call sigmoid,relu and return the list of the predicted output for the images.             
def prediced_output(weight1,weight11,weight2,weight22,Output_descriptor):
    Number_image,number_of_attribute=Output_descriptor.shape
    predict=[]
    for k in range(Number_image):
            x=Output_descriptor[k,:].reshape([1,-1])
            z=relu((x.dot(weight1)+weight11))

            y=sigmoid(z.dot(weight2)+weight22)

            predict.append(y)
    return predict


# In[60]:






# Used for the purpose of training the neural. It will return a list of descriptopr which is used as descriptior input for the neural traning.



def train():
# Calling Image Desxcriptor function for each image which consist of functions Histogram, Gradient Magnitude. It will return a Descriotor of each image.      
    descriptor_final_list=[]
    print("reluu for train images................................................................................")
    output_list=[]
    img=cv2.imread('crop001030c.bmp')
    image_descriptorr=images_descriptor(img,'crop001030c.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(1)
    img=cv2.imread('crop001034b.bmp')
    image_descriptorr=images_descriptor(img,'crop001034b.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(1)
    img=cv2.imread('crop001063b.bmp')
    image_descriptorr=images_descriptor(img,'crop001063b.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(1)
    img=cv2.imread('crop001070a.bmp')
    image_descriptorr=images_descriptor(img,'crop001070a.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(1)
    img=cv2.imread('crop001275b.bmp')
    image_descriptorr=images_descriptor(img,'crop001275b.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(1)
    img=cv2.imread('crop001278a.bmp')
    image_descriptorr=images_descriptor(img,'crop001278a.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(1)
    outputt1=np.zeros((len(image_descriptorr),1))
    print("1278")
    for ty in range(len(outputt1)):
        outputt1[ty][0]=image_descriptorr[ty]
        print(outputt1[ty][0])
    
    
    
    img=cv2.imread('crop001500b.bmp')
    image_descriptorr=images_descriptor(img,'crop001500b.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(1)
    img=cv2.imread('crop001672b.bmp')
    image_descriptorr=images_descriptor(img,'crop001672b.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(1)
    img=cv2.imread('person_and_bike_026a.bmp')
    image_descriptorr=images_descriptor(img,'person_and_bike_026a.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(1)
    img=cv2.imread('person_and_bike_151a.bmp')
    image_descriptorr=images_descriptor(img,'person_and_bike_151a.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(1)



    # negative images

    img=cv2.imread('00000003a_cut.bmp')
    image_descriptorr=images_descriptor(img,'00000003a_cut.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(0)

    img=cv2.imread('00000057a_cut.bmp')
    image_descriptorr=images_descriptor(img,'00000057a_cut.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(0)

    img=cv2.imread('00000090a_cut.bmp')
    image_descriptorr=images_descriptor(img,'00000090a_cut.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(0)


    img=cv2.imread('00000091a_cut.bmp')
    image_descriptorr=images_descriptor(img,'00000091a_cut.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(0)


    img=cv2.imread('00000118a_cut.bmp')
    image_descriptorr=images_descriptor(img,'00000118a_cut.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(0)

    img=cv2.imread('01-03e_cut.bmp')
    image_descriptorr=images_descriptor(img,'01-03e_cut.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(0)

    img=cv2.imread('no_person__no_bike_219_cut.bmp')
    image_descriptorr=images_descriptor(img,'no_person__no_bike_219_cut.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(0)


    img=cv2.imread('no_person__no_bike_258_Cut.bmp')
    image_descriptorr=images_descriptor(img,'no_person__no_bike_258_Cut.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(0)

    img=cv2.imread('no_person__no_bike_259_cut.bmp')
    image_descriptorr=images_descriptor(img,'no_person__no_bike_259_cut.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(0)

    img=cv2.imread('no_person__no_bike_264_cut.bmp')
    image_descriptorr=images_descriptor(img,'no_person__no_bike_264_cut.bmp')
    descriptor_final_list.append(image_descriptorr)
    output_list.append(0)



# Cnverting a list of Descriptor which consist of all Images descriptor into a Array which will then send as a input for the neural. 

    descriptor_array=np.zeros((20,len(descriptor_final_list[0])),dtype=float)
    length1,length2=descriptor_array.shape

    for x in range(length1):
        descriptor_array[x]=descriptor_final_list[x]

    
    output_array=np.zeros((20,1),dtype=float)
    length3,length4=output_array.shape
# Output Array which shows 1 if the iamge contain Human else 0 if not Human
    for x in range(length3):
        output_array[x][0]=output_list[x]
    
    return descriptor_array,output_array


# In[61]:


def main():

    descripted_array,outputed_array=train()
# After training we will get the weights which can then used for the purpose of testing on the images. These weights then used on the test images and help to predict the Hum

    weight1,weight11,weight2,weight22=neural(descripted_array,outputed_array,1000,0.01,100)
    
# Now we will test these weights on the test images.
    output_descriptor_array=[]
    out_predicated=[]
# test images
    img=cv2.imread('crop001008b.bmp')
    image_descriptorr=images_descriptor(img,'crop001008b.bmp')
    output_descriptor_array.append(image_descriptorr)
    out_predicated.append(1)
    
    img=cv2.imread('00000053a_cut.bmp')
    image_descriptorr=images_descriptor(img,'00000053a_cut.bmp')
    output_descriptor_array.append(image_descriptorr)
    out_predicated.append(0)
    
    img=cv2.imread('crop_000010b.bmp')
    image_descriptorr=images_descriptor(img,'crop_000010b.bmp')
    output_descriptor_array.append(image_descriptorr)
    out_predicated.append(1)
    
    img=cv2.imread('00000062a_cut.bmp')
    image_descriptorr=images_descriptor(img,'00000062a_cut.bmp')
    output_descriptor_array.append(image_descriptorr)
    out_predicated.append(0)
 
    img=cv2.imread('crop001028a.bmp')
    image_descriptorr=images_descriptor(img,'crop001028a.bmp')
    output_descriptor_array.append(image_descriptorr)
    out_predicated.append(1)
    
    img=cv2.imread('00000093a_cut.bmp')
    image_descriptorr=images_descriptor(img,'00000093a_cut.bmp')
    output_descriptor_array.append(image_descriptorr)
    out_predicated.append(0)

    img=cv2.imread('crop001045b.bmp')
    image_descriptorr=images_descriptor(img,'crop001045b.bmp')
    output_descriptor_array.append(image_descriptorr)
    out_predicated.append(1)
    outputt1=np.zeros((len(image_descriptorr),1))
    print("1045")
    for ty in range(len(outputt1)):
        outputt1[ty][0]=image_descriptorr[ty]
        print(outputt1[ty][0])
    
    
    img=cv2.imread('no_person__no_bike_213_cut.bmp')
    image_descriptorr=images_descriptor(img,'no_person__no_bike_213_cut.bmp')
    output_descriptor_array.append(image_descriptorr)
    out_predicated.append(0)

    img=cv2.imread('crop001047b.bmp')
    image_descriptorr=images_descriptor(img,'crop001047b.bmp')
    output_descriptor_array.append(image_descriptorr)
    out_predicated.append(1)
    
    img=cv2.imread('no_person__no_bike_247_cut.bmp')
    image_descriptorr=images_descriptor(img,'no_person__no_bike_247_cut.bmp')
    output_descriptor_array.append(image_descriptorr)
    out_predicated.append(0)
    
    
    out_descriptorrrrr=np.zeros((len(output_descriptor_array),len(output_descriptor_array[0])),dtype=float)
    for i in range(len(out_descriptorrrrr)):
        out_descriptorrrrr[i]=output_descriptor_array[i]
    predicce=prediced_output(weight1,weight11,weight2,weight22,out_descriptorrrrr)


# Create a list for the purpose of Matching for the Images. If it is 0 or less than 0.5 means there is no Human and Human for if it is greater than 0.5 or equal to 0.5.

    checkmatch=[]

    for k in range(len(predicce)):
        if(predicce[k]>=0.5):
            checkmatch.append(1)
        else:
            checkmatch.append(0)
    print(checkmatch)

   


# if __name__ == '__main__':
#     main()

# In[62]:


if __name__ == '__main__':
    main()
    


# In[ ]:




