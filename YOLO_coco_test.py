import numpy as np
import tensorflow as tf
import cv2
import time
import sys
import os 
import shutil
import cfg

def get_classes_names():
    names = []
    with open('coco_names.txt') as f:
        for name in f.readlines():
            name = name[:-1]
            names.append(name)
    return names

class YOLO_TF:
    def __init__(self,batch_size,input_size,threshold):
        self.weights_file = 'weights/yolo_weights.ckpt'

        self.alpha = 0.1
        self.threshold = threshold
        self.iou_threshold = 0.5

        self.classes = get_classes_names()
        self.input_size = input_size
        self.num_class = len(self.classes)
        self.boxes_per_cell = 5
        self.cell_size = self.input_size//32
        # self.batch_size = len(self.fromfile)
        self.batch_size = batch_size
        self.debug = {}

        self.anchors = np.array([[0.57273,0.677385],[1.87446,2.06253],[3.33843,5.47434],[7.88282,3.52778],[9.77052,9.16828]])
        self.offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),(self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.build_networks()
                
    def build_networks(self):
        print "Building YOLO graph..."
        self.x = tf.placeholder('float32',[self.batch_size,self.input_size,self.input_size,3])

        self.conv_0 = self.conv_layer_bn(0,self.x,32,3,1)
        self.pool_1 = self.pooling_layer(1,self.conv_0,2,2)
        self.conv_2 = self.conv_layer_bn(2,self.pool_1,64,3,1)
        self.pool_3 = self.pooling_layer(3,self.conv_2,2,2)
        self.conv_4 = self.conv_layer_bn(4,self.pool_3,128,3,1)
        self.conv_5 = self.conv_layer_bn(5,self.conv_4,64,1,1)
        self.conv_6 = self.conv_layer_bn(6,self.conv_5,128,3,1)
        self.pool_7 = self.pooling_layer(7,self.conv_6,2,2)

        self.conv_8 = self.conv_layer_bn(8,self.pool_7,256,3,1)
        self.conv_9 = self.conv_layer_bn(9,self.conv_8,128,1,1)
        self.conv_10 = self.conv_layer_bn(10,self.conv_9,256,3,1)
        self.pool_11 = self.pooling_layer(11,self.conv_10,2,2)

        self.conv_12 = self.conv_layer_bn(12,self.pool_11,512,3,1)
        self.conv_13 = self.conv_layer_bn(13,self.conv_12,256,1,1)
        self.conv_14 = self.conv_layer_bn(14,self.conv_13,512,3,1)
        self.conv_15 = self.conv_layer_bn(15,self.conv_14,256,1,1)
        self.conv_16 = self.conv_layer_bn(16,self.conv_15,512,3,1)
        self.pool_17 = self.pooling_layer(17,self.conv_16,2,2)

        self.conv_18 = self.conv_layer_bn(18,self.pool_17,1024,3,1)
        self.conv_19 = self.conv_layer_bn(19,self.conv_18,512,1,1)
        self.conv_20 = self.conv_layer_bn(20,self.conv_19,1024,3,1)
        self.conv_21 = self.conv_layer_bn(21,self.conv_20,512,1,1)
        self.conv_22 = self.conv_layer_bn(22,self.conv_21,1024,3,1)
        self.conv_23 = self.conv_layer_bn(23,self.conv_22,1024,3,1)
        self.conv_24 = self.conv_layer_bn(24,self.conv_23,1024,3,1)

        self.rout_25 = self.conv_16
        self.conv_26 = self.conv_layer_bn(26,self.rout_25,64,1,1)
        self.reor_27 = self.reorg_layer(27,self.conv_26)

        self.rout_28 = tf.concat([self.reor_27,self.conv_24], 3)
        print '28', self.rout_28.get_shape()
        # debug
        #temp = tf.transpose(self.rout_28, (0,3,1,2))
        #self.debug[28] = tf.reshape(temp, [-1, 1])

        self.conv_29 = self.conv_layer_bn(29,self.rout_28,1024,3,1)
        self.conv_30 = self.conv_layer_linear_ac(30,self.conv_29,425,1,1)# (x, 13,13,425)
        self.boxes, self.probs = self.output_layer(self.conv_30)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.variable_to_restore = tf.global_variables()

        # for name in self.variable_to_restore:
            # print name

        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.saver.restore(self.sess, self.weights_file)

        print "Loading complete!" + '\n'

    def reorg_layer(self,idx,inputs):
        # inputs (1,26,26,64)
        num = int(inputs.get_shape()[0])
        width = int(inputs.get_shape()[1])
        channel = int(inputs.get_shape()[3])# 64

        inputs =  tf.transpose(inputs, (0,3,1,2))# (1,64,26,26)

        reorg_filter_0 = np.tile(np.expand_dims(np.tile(np.array([[1,0],[1,0],[0,0],[0,0]]), (width*channel/4,width/2)),0),(num,1,1,1))# (1,1,1664,26) 1664=26*64
        reorg_filter_1 = np.tile(np.expand_dims(np.tile(np.array([[0,1],[0,1],[0,0],[0,0]]), (width*channel/4,width/2)),0),(num,1,1,1))
        reorg_filter_2 = np.tile(np.expand_dims(np.tile(np.array([[0,0],[0,0],[1,0],[1,0]]), (width*channel/4,width/2)),0),(num,1,1,1))
        reorg_filter_3 = np.tile(np.expand_dims(np.tile(np.array([[0,0],[0,0],[0,1],[0,1]]), (width*channel/4,width/2)),0),(num,1,1,1))

        reorg_filter_0 = reorg_filter_0==1# (1,1,1664,26) 1664=26*64
        reorg_filter_1 = reorg_filter_1==1
        reorg_filter_2 = reorg_filter_2==1
        reorg_filter_3 = reorg_filter_3==1

        inputs_reshaped = tf.reshape(inputs, (num,1,width*channel,width))# (1,1,1664,26) 1664=26*64

        returnn_0 = tf.boolean_mask(inputs_reshaped, reorg_filter_0)# (1,1,832,13) 83213*64
        returnn_1 = tf.boolean_mask(inputs_reshaped, reorg_filter_1)
        returnn_2 = tf.boolean_mask(inputs_reshaped, reorg_filter_2)
        returnn_3 = tf.boolean_mask(inputs_reshaped, reorg_filter_3)

        returnn_0 = tf.reshape(returnn_0, (num,channel,width/2,width/2))# (1,64,13,13)
        returnn_1 = tf.reshape(returnn_1, (num,channel,width/2,width/2))
        returnn_2 = tf.reshape(returnn_2, (num,channel,width/2,width/2))
        returnn_3 = tf.reshape(returnn_3, (num,channel,width/2,width/2))

        returnn = tf.transpose(tf.concat([returnn_0,returnn_1,returnn_2,returnn_3], 1), (0,2,3,1))# (1,13,13,256)
        
        # debug
        # temp = tf.transpose(returnn, (0,3,1,2))
        # self.debug[idx] = tf.reshape(temp, [-1, 1])
        # self.debug[idx] = returnn[0,0,0,:]

        print idx, returnn.get_shape()
        return returnn

    def conv_layer_linear_ac(self,idx,inputs,filters,size,stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1),name='weight')
        biases = tf.Variable(tf.constant(0.1, shape=[filters]),name='biases')

        pad_size = size//2
        pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        inputs_pad = tf.pad(inputs,pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')  
        conv_biased = tf.add(conv,biases,name=str(idx)+'_conv_biased')  
        returnn = tf.maximum(conv_biased,conv_biased,name=str(idx)+'_linear_relu')
        print idx, returnn.get_shape()

        # debug
        # temp = tf.transpose(returnn, (0,3,1,2))
        # self.debug[idx] = tf.reshape(temp, [-1, 1])
        # self.debug[99] = biases

        return returnn

    def conv_layer_bn(self,idx,inputs,filters,size,stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.truncated_normal([size,size,int(channels),filters], stddev=0.1),name='weight')
        biases = tf.Variable(tf.constant(0.1, shape=[filters]),name='biases')

        pad_size = size//2
        pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        inputs_pad = tf.pad(inputs,pad_mat)
        
        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')

        # batch normalization
        depth = conv.get_shape()[3]

        scale = tf.Variable(np.ones([depth,], dtype='float32'),name='scale')
        shift = tf.Variable(np.zeros([depth,], dtype='float32'),name='shift')
        BN_EPSILON = 0.00001

        mean = tf.Variable(np.ones([depth,], dtype='float32'),name='rolling_mean')
        variance = tf.Variable(np.ones([depth,], dtype='float32'),name='rolling_variance')
        
        conv_bn = tf.nn.batch_normalization(conv, mean, variance, shift, scale, BN_EPSILON)
        conv_biased = tf.add(conv_bn,biases,name=str(idx)+'_conv_biased')  
        returnn = tf.maximum(self.alpha*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')
        
        # debug
        # temp = tf.transpose(returnn, (0,3,1,2))
        # self.debug[idx] = tf.reshape(temp, [-1, 1])
        # self.debug[idx] = returnn[0,0,0,:]

        print idx, returnn.get_shape()
        return returnn

    def pooling_layer(self,idx,inputs,size,stride):
        returnn = tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')
        print idx, returnn.get_shape()
        return returnn

    def output_layer(self,predicts):
        # predicts (?,13,13,425)
        # self.debug = predicts[0][:][0][0]

        predicts = tf.reshape(predicts, [self.batch_size,self.cell_size,self.cell_size,self.boxes_per_cell,5+self.num_class])# (?,13,13,5,85)
        predict_classes = tf.stack([predicts[:,:,:,0,5:],predicts[:,:,:,1,5:],predicts[:,:,:,2,5:],predicts[:,:,:,3,5:],predicts[:,:,:,4,5:]])# (5,?,13,13,80)
        predict_classes = tf.transpose(predict_classes,(1,2,3,0,4))# (?,13,13,5,80)
        predict_classes = tf.exp(predict_classes)/tf.tile(tf.expand_dims(tf.reduce_sum(tf.exp(predict_classes),axis=4),4),(1,1,1,1,self.num_class))# (?,13,13,5,80) softmax
        
        predict_scales = tf.stack([predicts[:,:,:,0,4],predicts[:,:,:,1,4],predicts[:,:,:,2,4],predicts[:,:,:,3,4],predicts[:,:,:,4,4]])# (5,?,13,13)
        predict_scales = tf.transpose(predict_scales,(1,2,3,0))# (?,13,13,5)
        predict_scales = 1.0/(1+tf.exp(-1.0*predict_scales))# (?,13,13,5) logistic
        predict_scales = tf.tile(tf.expand_dims(predict_scales, 4), (1,1,1,1,self.num_class))# (?,13,13,5,5)

        probs = predict_classes * predict_scales# (?,13,13,5,5)
        # probs = tf.reduce_max(probs,4)

        predict_boxes = tf.stack([predicts[:,:,:,0,0:4],predicts[:,:,:,1,0:4],predicts[:,:,:,2,0:4],predicts[:,:,:,3,0:4],predicts[:,:,:,4,0:4]])# (5,?,13,13,4)
        predict_boxes = tf.transpose(predict_boxes,(1,2,3,0,4))# (?,13,13,5,4)

        anchors = tf.constant(self.anchors, dtype=tf.float32)
        anchors_w = tf.tile(tf.reshape(anchors[:,0],(1,1,1,self.boxes_per_cell)),(self.batch_size,self.cell_size,self.cell_size,1))# (?,13,13,5)
        anchors_h = tf.tile(tf.reshape(anchors[:,1],(1,1,1,self.boxes_per_cell)),(self.batch_size,self.cell_size,self.cell_size,1))# (?,13,13,5)
        offset = tf.constant(self.offset, dtype=tf.float32)# (13,13,5)
        offset = tf.reshape(offset,[1, self.cell_size, self.cell_size, self.boxes_per_cell])# (1,13,13,5)
        offset = tf.tile(offset, [self.batch_size, 1, 1, 1]) # (?,13,13,5)
        predict_boxes_tran = tf.stack([(1.0/(1.0+tf.exp(-1.0*predict_boxes[:,:,:,:,0])) + offset) / self.cell_size,
                                       (1.0/(1.0+tf.exp(-1.0*predict_boxes[:,:,:,:,1])) + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
                                       anchors_w*tf.exp(predict_boxes[:,:,:,:,2]) / self.cell_size,
                                       anchors_h*tf.exp(predict_boxes[:,:,:,:,3]) / self.cell_size])# (4,?,13,13,5)
        predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])# (?,13,13,5,4) iou

        return predict_boxes_tran, probs

    def softmax(self,output):
        return np.exp(output)/np.sum(np.exp(output),axis=0)

    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def detect_from_pic(yolo,img,fill_num=0):

    pic_num = len(img)

    h_img = np.zeros(pic_num)
    w_img = np.zeros(pic_num)
    inputs = np.zeros((pic_num,yolo.input_size,yolo.input_size,3),dtype='float32')# (x,input_size,input_size,3)

    for i in range(pic_num):
        h_img[i],w_img[i],_ = img[i].shape
        img_RGB = cv2.cvtColor(img[i],cv2.COLOR_BGR2RGB)

        w = yolo.input_size
        h = yolo.input_size
        if (1.0 * w/w_img[i]) < (1.0 * h/h_img[i]):
            new_w = w
            new_h = (h_img[i] * w)/w_img[i]
        else:
            new_h = h;
            new_w = (w_img[i] * h)/h_img[i]
        
        new_w = int(new_w)
        new_h = int(new_h)

        resized = cv2.resize(img_RGB, (new_w, new_h))
        embeded = np.ones_like(np.zeros([h, w ,3])) * 127

        dx = (w - new_w)/2
        dy = (h - new_h)/2

        embeded[dy:dy+new_h,dx:dx+new_w,:] = resized

        img_resized = embeded
        img_resized_np = np.asarray(img_resized)
        
        inputs[i] = img_resized_np/255.0# (x,input_size,input_size,3)
        # debug
        # inputs[i] = img_RGB/255.0

    s = time.time()
    boxes, probs = yolo.sess.run([yolo.boxes, yolo.probs], feed_dict={yolo.x: inputs})

    # debug, boxes, probs = yolo.sess.run([yolo.debug, yolo.boxes, yolo.probs], feed_dict={yolo.x: inputs})
    # for (idx, output) in debug.items():
        # print idx
        # print output
        # filename = 'netoutput/' + str(idx) + '_output.txt'
        # np.savetxt(filename, output)

    total_time = round(time.time()-s,3)
    # print 'total time:',total_time,' secs'
    # print 'single time:',round(1.0*total_time/pic_num,3),' secs'

    # boxes (18,13,13,5,4)
    # probs (18,13,13,5,5)
    
    if fill_num > 0:
        boxes = boxes[:len(inputs)-fill_num]
        probs = probs[:len(inputs)-fill_num]
        inputs = inputs[:len(inputs)-fill_num]

    img_set = interpret_output(yolo, boxes, probs, inputs)
    # self.show_results(img,self.result)
    # self.show_results(inputs,self.result)
    return img_set

def interpret_output(yolo, boxes, probs, images):
    # boxes  (18,13,13,5,4)
    # probs  (18,13,13,5,80)
    # images (18,416,416,3)
    filter_mat_probs = np.array(probs>=yolo.threshold,dtype='bool')# (18,13,13,5,80)
    img_set = []

    for n in range(np.shape(images)[0]):
        filter_mat_boxes = np.nonzero(filter_mat_probs[n])# (4,x)
        boxes_filtered = boxes[n,filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]# (x,4)
        probs_filtered = probs[n,filter_mat_probs[n]]# (x,)
        classes_num_filtered = np.argmax(filter_mat_probs[n],axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]# (x,)
        
        argsort = np.array(np.argsort(probs_filtered))[::-1]# (x,)
        boxes_filtered = boxes_filtered[argsort]# (x,4)
        probs_filtered = probs_filtered[argsort]# (x,)
        classes_num_filtered = classes_num_filtered[argsort]# (x,)
        
        # nms
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0 : continue
            for j in range(i+1,len(boxes_filtered)):
                if iou(boxes_filtered[i], boxes_filtered[j]) > yolo.iou_threshold : 
                    probs_filtered[j] = 0.0
        
        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]# (xx,4)
        probs_filtered = probs_filtered[filter_iou]# (xx,)
        classes_num_filtered = classes_num_filtered[filter_iou]# (xx,)
        # print 'predict_nums:', len(boxes_filtered)

        img = show(yolo, images[n], classes_num_filtered, boxes_filtered, probs_filtered, n)
        img_set.append(img)
    return img_set
        
def show(yolo, image, pred_class, pred_boxes, probs_filtered, n):

    pred_boxes *= yolo.input_size
    img_cp = image.copy()
    img_cp = np.array(img_cp, dtype=np.float32)
    img_cp = cv2.cvtColor(img_cp,cv2.COLOR_RGB2BGR)

    class_name = yolo.classes

    for i in xrange(len(pred_boxes)):
        x = int(pred_boxes[i][0])
        y = int(pred_boxes[i][1])
        w = int(pred_boxes[i][2])//2
        h = int(pred_boxes[i][3])//2

        cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(0,0,0),-1)
        cv2.putText(img_cp,str(class_name[pred_class[i]]),(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        # cv2.putText(img_cp,str(class_name[pred_class[i]]) + ' ' +str(round(probs_filtered[i], 2)),(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

    return img_cp

def iou(box1,box2):
    tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
    lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
    if tb < 0 or lr < 0 : intersection = 0
    else : intersection =  tb*lr
    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def get_pic_from_path(file_path):
    fromfile = []
    for i in os.walk(file_path):
        for picfile in i[2]:
            fromfile.append(os.path.join(i[0], picfile))

    img = []
    for single_pic in fromfile:
        img.append(cv2.imread(single_pic))

    return img

def get_pic_from_video(filename):
    img_set = []
    vc = cv2.VideoCapture(filename)

    if vc.isOpened():
        rval , frame = vc.read()
    else: 
        print 'fuck'
        rval = False
      
    while True:
        rval, frame = vc.read()
        if rval == False:
            break
        img_set.append(frame)
        cv2.waitKey(1)
    vc.release()
    return img_set

def make_video(img_set,input_size):
    files = os.listdir('.')
    if 'temp' not in files:
        os.mkdir('temp')

    for i in range(len(img_set)):
        img = img_set[i]*255.0
        cv2.imwrite('temp/' + str(i) + '.jpg',img)

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('save_video.avi',fourcc,fps,(input_size,input_size))

    for i in range(len(img_set)):
        # frame = img_set[i]*255.0
        frame = cv2.imread('temp/' + str(i) + '.jpg')
        videoWriter.write(frame)
    videoWriter.release()
    shutil.rmtree('temp')

def show_pic(img_set,save_picture,show_picture):
    for i in range(len(img_set)):
        if save_picture:
            img = img_set[i]*255.0
            cv2.imwrite('outputs/'+str(i) + '.jpg',img)
        if show_picture:
            cv2.imshow('pic%d'%i,img_set[i])
            cv2.waitKey(0)

def main():
    # configure
    batch_size = cfg.batch_size
    input_size = cfg.input_size
    threshold = cfg.threshold
    source_file = cfg.source_file
    video_name = cfg.video_name
    save_picture = cfg.save_picture
    show_picture = cfg.show_picture

    # start
    if source_file == 'image':
        img_set = get_pic_from_path('test_pic/')
    elif source_file == 'video':
        img_set = get_pic_from_video('test_video/' + video_name)
    else:
        print "Error: source_file must be 'image' or 'video' !"
        os._exit()

    img_set_out = []

    if len(img_set) <= batch_size:
        yolo = YOLO_TF(len(img_set),input_size,threshold)
        img_set_out = detect_from_pic(yolo,img_set)
    else:
        yolo = YOLO_TF(batch_size,input_size,threshold)
        for i in range(len(img_set)//batch_size):
            sub_img = img_set[i*batch_size:(i+1)*batch_size]
            img_set_out = img_set_out + detect_from_pic(yolo,sub_img)

        empty_img = sub_img[0].copy()
        sub_img = img_set[len(img_set_out):]
        fill_num = batch_size-len(img_set)+len(img_set_out)
        for i in range(fill_num):
            sub_img.append(empty_img)

        sub_out = detect_from_pic(yolo,sub_img,fill_num)
        img_set_out = img_set_out + sub_out

    if source_file == 'image':
        show_pic(img_set_out,save_picture,show_picture)
    else:
        make_video(img_set_out,input_size)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    main()