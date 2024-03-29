#!/home/tops/bin/python
# -*- coding: utf-8 -*-
# vim:ts=4:sts=4:sw=4:et:fenc=utf8
import tensorflow as tf
from scheduler.distributed_scheduler import DistributedScheduler

tf.app.flags.DEFINE_string("model_name","HimModel","model")
tf.app.flags.DEFINE_integer("batch_size", 256, "batch size")
tf.app.flags.DEFINE_integer("predict_batch_size", 256, "predict batch size")
tf.app.flags.DEFINE_string("transform_json", "conf_fc_him.json", "feature transform config json file name")
tf.app.flags.DEFINE_integer("summary_step", 100, "how many steps summary written")
tf.app.flags.DEFINE_integer("print_step", 100, "how many steps print info")
tf.app.flags.DEFINE_integer("train_step", 200000, "train step of running.")
tf.app.flags.DEFINE_integer("checkpoint_step", 100, "how many steps summary written")
tf.app.flags.DEFINE_integer("checkpoint_sec", 300, "checkpoint save seconds.")
tf.app.flags.DEFINE_float("dnn_lr", 0.001, "learning rate for deep part")
tf.app.flags.DEFINE_integer("lr_decay_steps", 400000, "learning rate decay steps")
tf.app.flags.DEFINE_float("lr_decay_rate", 0.99, "learning rate decay rate")
tf.app.flags.DEFINE_float("dnn_l1_weight", 0.0, "l1 regularization weight for deep part")
tf.app.flags.DEFINE_float("dnn_l2_weight", 0.0, "l2 regularization weight for deep part")
tf.app.flags.DEFINE_float("cart_loss_weight", 0.0, "loss weight for cart sample")
tf.app.flags.DEFINE_float("pay_loss_weight", 0.0, "loss weight for pay sample")
tf.app.flags.DEFINE_string("item_hidden_units", "128,128,32", "")
tf.app.flags.DEFINE_string("user_hidden_units", "128,128,32", "")
tf.app.flags.DEFINE_string("dnn_activation", "leaky_relu", "")
tf.app.flags.DEFINE_integer("num_epoch", 3, "epoch used of training data")
tf.app.flags.DEFINE_float("lr_min",0.001,"lr_min")
tf.app.flags.DEFINE_string("lr_op","lr","lr_op")
tf.app.flags.DEFINE_integer("ubb_pos_slice_len",50,"ubb_pos_slice_len")
tf.app.flags.DEFINE_integer("attention_num_units",32,"attention_num_units")
tf.app.flags.DEFINE_integer("attention_num_output_units",32,"attention_num_output_units")
tf.app.flags.DEFINE_string("attention_num_units_forward",'64,32',"")
tf.app.flags.DEFINE_float("drop_out",None,"drop_out")
tf.app.flags.DEFINE_string("ubb_combiner", "max,sum", "ubb_combiner")
tf.app.flags.DEFINE_integer("ubb_dim", 32, "ubb_dim")
tf.app.flags.DEFINE_integer("target_comp_dim",8,"target_comp_dim")
tf.app.flags.DEFINE_integer("target_comp_num",4,"target_comp_num")
tf.app.flags.DEFINE_integer("user_emb_num",16,"user_emb_num")
tf.app.flags.DEFINE_integer("gru_group_num_units",16,"gru_group_num_units")
tf.app.flags.DEFINE_float("rate_for_uuloss",0.0005,"rate_for_uuloss")
tf.app.flags.DEFINE_float("rate_for_recloss",0.0005,"rate_for_recloss")
tf.app.flags.DEFINE_integer("group_embedding_num",128,"group_embedding_num")
tf.app.flags.DEFINE_integer("groups_num",5,"groups_num")
tf.app.flags.DEFINE_string("clip_gradients", "true", "true/false")
tf.app.flags.DEFINE_string("checkpoint_dir", "../ctr/test/", "checkpoint dir")
tf.app.flags.DEFINE_string("running_mode", "train", "running mode: train/predict")
tf.app.flags.DEFINE_integer("logits_dimension", 1, "final logits dimension")
tf.app.flags.DEFINE_float("ema_decay", 0.999, "exponential moving average decay for loss")
tf.app.flags.DEFINE_string("decision_layer_units","1024,512,128","decision_layer_units")
tf.app.flags.DEFINE_string("model_num",None,"model_num")

tf.app.flags.DEFINE_string("group_forward_layer_units","256,128","group_forward_layer_units")
tf.app.flags.DEFINE_float("temperature",0.01,"temperature")
tf.app.flags.DEFINE_string("encoder_num_units_forward",'64,32',"encoder_num_units_forward")
tf.app.flags.DEFINE_string("decoder_num_units_forward",'64,32',"decoder_num_units_forward")

tf.app.flags.DEFINE_integer("embedding_for_att",512,"embedding_for_att")
tf.app.flags.DEFINE_integer("embedding_for_att_unit",256,"embedding_for_att_unit")

FLAGS = tf.app.flags.FLAGS


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    scheduler = DistributedScheduler()

    if FLAGS.running_mode == "train":
        scheduler.train(FLAGS)
    elif FLAGS.running_mode == "predict":
        scheduler.eval(FLAGS)
    else:
        print("bad running mode: %s" % FLAGS.running_mode)
    return


if __name__ == '__main__':
    tf.app.run()
