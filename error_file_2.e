/home/ksuresh/.conda/envs/py27t04/lib/python2.7/site-packages/torch/nn/functional.py:1749: UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
  "See the documentation of nn.Upsample for details.".format(mode))
/home/ksuresh/fpn.pytorch-master/lib/model/rpn/rpn_fpn.py:79: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
  rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape)
/home/ksuresh/fpn.pytorch-master/lib/model/fpn/fpn.py:255: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
  cls_prob = F.softmax(cls_score)
trainval_net.py:336: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  loss_temp += loss.data[0]
trainval_net.py:349: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  loss_rpn_cls = rpn_loss_cls.mean().data[0]
trainval_net.py:350: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  loss_rpn_box = rpn_loss_box.mean().data[0]
trainval_net.py:351: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
trainval_net.py:352: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
