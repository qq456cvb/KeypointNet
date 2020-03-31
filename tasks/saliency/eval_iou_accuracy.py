import numpy as np
import glob
import os

def load_obj(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()
    data = np.zeros((  len(lines), 4 ))
    for idx in range(len(lines)):
        line = lines[idx]
        line = line.split()[1:]
        line = [float(i) for i in line]
        data[idx, 0], data[idx,1], data[idx,2], data[idx,3] = line[0], line[1], line[2], line[-1]
    return data

root = 'results'

pred_filenames = glob.glob(os.path.join(root, '*_pred.obj'))
gt_filenames = [f.rstrip('_pred.obj') + '_gt.obj' for f in pred_filenames]

num_room = len(gt_filenames)


gt_classes = [0 for _ in range(13)]
positive_classes = [0 for _ in range(13)]
true_positive_classes = [0 for _ in range(13)]
for i in range(num_room):
    print(i)
    pred_label = load_obj(pred_filenames[i])[:,-1]
    gt_label = load_obj( gt_filenames[i] )[:,-1]
    assert len(pred_label) == len(gt_label)
    print(gt_label.shape)
    for j in range(gt_label.shape[0]):
        gt_l = int(gt_label[j])
        pred_l = int(pred_label[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l==pred_l)


print(gt_classes)
print(positive_classes)
print(true_positive_classes)


oa = sum(true_positive_classes) /float(sum(positive_classes))
print('Overall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))
meanAcc = 0
for tp, gt in zip(true_positive_classes, gt_classes):
    meanAcc += ( tp / float(gt) )

meanAcc /= 13
print('Mean accuracy: {0}'.format(  meanAcc    ) )

print('IoU:')
iou_list = []
for i in range(13):
    iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
    print(iou)
    iou_list.append(iou)

print(sum(iou_list)/13.0)
meanIOU = sum(iou_list)/13.0

with open('test_log.txt', 'a') as f:
    f.write( ' OA {:.5f} MA {:.5f} MIOU {:.5f} \n'.format(  oa, meanAcc,  meanIOU ) )


