# coding=utf-8


FEATURE_GROUPS = ["item", "user_profile", "user_seq", "ubb_id", "ubb_value", "context", "seq_length"]


DTYPE_ATTR_NAME = 'dtype'
COMBINER_ATTR_NAME = 'combiner'
HASH_BUCKET_SIZE_ATTR_NAME = 'hash_bucket_size'
DIMENSION_ATTR_NAME = 'dimension'
BUCKET_SIZE_ATTR_NAME = 'bucket_size'
KEYS_ATTR_NAME = 'keys'
BOUNDARIES_ATTR_NAME = 'boundaries'
VOCABULARY_FILE_ATTR_NAME = 'vocabulary_file'
INITIALIZER_ATTR_NAME = 'initializer'
NORMALIZER_ATTR_NAME = 'normalizer'
CHECKPOINT_TO_LOAD_ATTR_NAME = 'checkpoint_to_load'
TENSOR_IN_CKPT_ATTR_NAME = 'tensor_in_ckpt'
TRAINABLE_ATTR_NAME = 'trainable'
MAX_NORM_ATTR_NAME = 'max_norm'
DEFAULT_VALUE_ATTR_NAME = 'default_value'
INPUT_NAME = 'input_feature_name'
OUTPUT_NAME = 'output_feature_name'
TRANSFORM_NAME = 'transform_name'
FEATURE_INDEX = 'feature_index'
FEATURE_USED = 'to'  # 'wide', 'deep'
FEATURE_USED_WIDE = "wide"
FEATURE_USED_DEEP = "deep"
FEATURE_USED_LENGTH = "length"
FEATURE_USED_SEQ = "seq"
FEATURE_ITEM_REAL  = "item_real"
VOCABULARY_KEYS_ATTR_NAME = 'vocabulary_keys'
SEQ_FEATURES = "seq_features"
SHARE_NAME = "shared_name"
LENGTH = "length"
MAX_LENGTH = "max_length"
SPLIT_CHAR_KK = "split_char_kk"
SPLIT_CHAR_KV = "split_char_kv"

class ModeKeys(object):
  """Standard names for model modes.

  The following standard keys are defined:

  * `TRAIN`: training mode.
  * `EVAL`: evaluation mode.
  * `PREDICT`: inference mode.
  """

  TRAIN = 'train'
  EVAL = 'eval'
  PREDICT = 'infer'