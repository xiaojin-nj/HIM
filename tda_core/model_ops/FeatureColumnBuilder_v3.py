#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
from tensorflow.python.ops import variable_scope
import six


# feature transform constants
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
FEATURE_USED_DEEP_ID = "deep_id"
FEATURE_USED_GATE = "gate"
FEATURE_USED_LENGTH = "length"
FEATURE_USED_SEQ = "seq"
FEATURE_USED_UBB = "ubb"
FEATURE_USED_UBB_NEG = "ubb_neg"
FEATURE_ITEM_REAL  = "item_real"
FEATURE_USED_ATT = "hard_att"
VOCABULARY_KEYS_ATTR_NAME = 'vocabulary_keys'
SEQ_FEATURES = "seq_features"
SHARE_NAME = "shared_name"
LENGTH = "length"
MAX_LENGTH = "max_length"
SPLIT_CHAR_KK = "split_char_kk"
SPLIT_CHAR_KV = "split_char_kv"
FEATURE_USED_GROUP = "group"
FEATURE_GROUPS = {
                    "deep" : "item",
                    "wide" : "user_profile",
                    "length" : "seq_length",
                    "seq" : "user_seq",
                    "item_real" : "context",
                    "ubb" : "ubb",
                    "hard_att" : "hard_att"
                  }

# parsing from json and store transform parameters
class FeatureColumnConfig(object):
    def __init__(self, config):
        self._configDict = {}
        if INPUT_NAME not in config or OUTPUT_NAME not in config or TRANSFORM_NAME not in config:
            print('invalid feature transform: %s' % config)
            raise Exception('invalid feature transform config')

        for key, value in config.items():
            if key == DTYPE_ATTR_NAME:
                if value == 'int32':
                    self._configDict[DTYPE_ATTR_NAME] = tf.int32
                elif value == 'int64':
                    self._configDict[DTYPE_ATTR_NAME] = tf.int64
                elif value == 'string':
                    self._configDict[DTYPE_ATTR_NAME] = tf.string
                elif value == 'float32':
                    self._configDict[DTYPE_ATTR_NAME] = tf.float32
                elif value == 'float64':
                    self._configDict[DTYPE_ATTR_NAME] = tf.float64
                continue
            # other configs
            self._configDict[key] = value

    def get(self, key):
        return self._configDict.get(key)


# looping on configured features and transform them.
class FeatureColumnBuilder(object):

    def __init__(self, configDict, flags, features):
        self._featureColumnDict = {}  # feature name -> feature column
        #TODO
        # add  dict {key: feature_group_name, value: Colum_list
        self._deepColumnList = []  # only deep column features are added.
        self._wideColumnList = []
        self._groupColumnList = []
        self._seqColumnList = []
        self._flags = flags
        self._featureColumnConfigList = []
        self._seqColumnDict = {}
        self._seqnormDict = {}
        self._lengthColumnList = []
        self._itemRealColumList = []
        self._itemRealDict = {}
        self._features = features
        self._ubb_embedding_value = []
        self._ubb_embedding_id = []
        self._feature_dict = {}
        self._hardAttColumnList = []
        self._itemIdColumnList = []
        self._ubb_embedding_value_neg = []
        self._ubb_embedding_id_neg = []
        self._gate_judge = []

        # transform functions
        self._transformFunctions = {
            "sparse_hash": "sparse_column_with_hash_bucket",
            "sparse_keys": "sparse_column_with_keys",
            "sparse_vocabulary": "sparse_column_with_vocabulary_file",
            "sparse_integerized": "sparse_column_with_integerized_feature",
            "embedding": "embedding_column",
            "shared_embedding": "shared_embedding_columns",
            "one_hot": "one_hot_column",
            "bucketized": "bucketized_column",
            "real_value": "real_valued_column",
            "crossed": "crossed_column",
            "seq_embedding": "seq_embedding",
            "bucketized_embedding":"bucketized_embedding",
            "string_value":"string_value"
        }

        # load config
        output2input = {}
        parseSuccess = True
        for fc in configDict['feature_columns']:
            try:
                featureConfig = FeatureColumnConfig(fc)
                self._featureColumnConfigList.append(featureConfig)
                if featureConfig.get(OUTPUT_NAME) in output2input:
                    print("duplicate output feature: %s" % featureConfig.get(OUTPUT_NAME))
                    parseSuccess = False
                output2input[featureConfig.get(OUTPUT_NAME)] = featureConfig.get(INPUT_NAME)
            except:
                parseSuccess = False
        if not parseSuccess:
            raise Exception('load and parse config json file failed')

    def _getFeatureKey(self, key):
        if key in FEATURE_GROUPS:
            return FEATURE_GROUPS[key]
        return key

    def getInputTensorConf(self):
        # get input tensors
        outputSet = set()
        for fc in self._featureColumnConfigList:
            if fc.get(INPUT_NAME) != fc.get(OUTPUT_NAME):
                outputSet.add(fc.get(OUTPUT_NAME))
        inputTensorConfList = []
        for fc in self._featureColumnConfigList:
            if fc.get(INPUT_NAME) == fc.get(OUTPUT_NAME) or fc.get(INPUT_NAME) not in outputSet:
                if ',' not in fc.get(INPUT_NAME):
                    inputTensorConfList.append({'name':fc.get(INPUT_NAME), 'dtype':fc.get(DTYPE_ATTR_NAME)})
        print("---- input tensors ----")
        print(inputTensorConfList)
        return inputTensorConfList

    def RealValuedColumn(self, featureColumnConfig):
        default_val = featureColumnConfig.get(DEFAULT_VALUE_ATTR_NAME)
        if default_val is None:
            default_val = 0.0

        normalizer = featureColumnConfig.get(NORMALIZER_ATTR_NAME)
        normalizer_fn = None
        if normalizer == 'z-score':
            stddev_val = featureColumnConfig.get('stddev')
            mean_val = featureColumnConfig.get('mean')
            if stddev_val is not None and mean_val is not None:
                normalizer_fn = lambda x, mean = mean_val, stddev = stddev_val: (x - mean) / (stddev + 1e-6)

        column = tf.contrib.layers.real_valued_column(
            featureColumnConfig.get(INPUT_NAME),
            dtype=featureColumnConfig.get(DTYPE_ATTR_NAME),
            default_value=default_val,
            dimension=featureColumnConfig.get(DIMENSION_ATTR_NAME),
            normalizer=normalizer_fn)

        return column

    def SharedEmbeddingColumns(self, featureColumnConfig):
        hash_id = tf.contrib.layers.sparse_column_with_hash_bucket(
            column_name=featureColumnConfig.get(INPUT_NAME),
            hash_bucket_size=featureColumnConfig.get(BUCKET_SIZE_ATTR_NAME),
            dtype=featureColumnConfig.get(DTYPE_ATTR_NAME),
            combiner=featureColumnConfig.get(COMBINER_ATTR_NAME))

        seq_idFeatureColumnList_1 = []
        seq_idFeatureColumnList_1.append(hash_id)
        ckptName = None
        tensorName = None
        outColumn = tf.contrib.layers.shared_embedding_columns(
            sparse_id_columns=seq_idFeatureColumnList_1,
            dimension=featureColumnConfig.get(DIMENSION_ATTR_NAME),
            combiner=featureColumnConfig.get(COMBINER_ATTR_NAME),
            shared_embedding_name=featureColumnConfig.get(SHARE_NAME),
            initializer=tf.contrib.layers.xavier_initializer(),
            ckpt_to_load_from=ckptName,
            tensor_name_in_ckpt=tensorName,
            max_norm=featureColumnConfig.get(MAX_NORM_ATTR_NAME),
            trainable=featureColumnConfig.get(TRAINABLE_ATTR_NAME))

        self._featureColumnDict[featureColumnConfig.get(OUTPUT_NAME)] = outColumn
        return outColumn[0]

    def StringValueColumn(self, fc):
        with variable_scope.variable_scope(fc.get(INPUT_NAME)+'input_layer', values=tuple(six.itervalues(self._features))):

            default_val = fc.get(DEFAULT_VALUE_ATTR_NAME)
            if default_val is None:
                default_val = 0.0
            feature_value = tf.contrib.layers.real_valued_column(
                fc.get(INPUT_NAME) + '_value',
                dtype=tf.float32,
                default_value=default_val,
                dimension=1,
                normalizer=None)

            hash_id = tf.contrib.layers.sparse_column_with_hash_bucket(
                column_name=fc.get(INPUT_NAME) + '_id',
                hash_bucket_size=fc.get(BUCKET_SIZE_ATTR_NAME),
                dtype=fc.get(DTYPE_ATTR_NAME),
                combiner='mean')
            seq_idFeatureColumnList_1 = []
            seq_idFeatureColumnList_1.append(hash_id)
            ckptName = None
            tensorName = None
            feature_emb = tf.contrib.layers.shared_embedding_columns(
                sparse_id_columns=seq_idFeatureColumnList_1,
                dimension=fc.get(DIMENSION_ATTR_NAME),
                combiner='mean',
                shared_embedding_name=fc.get(SHARE_NAME),
                initializer=tf.contrib.layers.xavier_initializer(),
                ckpt_to_load_from=ckptName,
                tensor_name_in_ckpt=tensorName,
                max_norm=fc.get(MAX_NORM_ATTR_NAME),
                trainable=fc.get(TRAINABLE_ATTR_NAME))

            if fc.get(FEATURE_USED) == FEATURE_USED_UBB:
                self._ubb_embedding_id.append(feature_emb[0])
                self._ubb_embedding_value.append(feature_value)
            elif fc.get(FEATURE_USED) == FEATURE_USED_UBB_NEG:
                self._ubb_embedding_id_neg.append(feature_emb[0])
                self._ubb_embedding_value_neg.append(feature_value)
        return [feature_emb[0], feature_value]

    def buildColumns(self):
        for fc in self._featureColumnConfigList:
            self.buildColumn(fc)
        return

    def buildColumn(self, fc):
        transformName = fc.get(TRANSFORM_NAME)
        column = None
        if transformName == self._transformFunctions["shared_embedding"]:
            column = self.SharedEmbeddingColumns(fc)
        elif transformName == self._transformFunctions["real_value"]:
            column = self.RealValuedColumn(fc)
        elif transformName == self._transformFunctions["string_value"]:
            column = self.StringValueColumn(fc)
        else:
            raise Exception("not supported feature transform: %s, feature: %s, feature index: %d" % (
                transformName, fc.get(INPUT_NAME), fc.get(FEATURE_INDEX)))
        if column:
            self._featureColumnDict[fc.get(OUTPUT_NAME)] = column
            if fc.get(FEATURE_USED) == FEATURE_USED_WIDE:
                self._wideColumnList.append(column)
            elif fc.get(FEATURE_USED) == FEATURE_USED_DEEP:
                self._deepColumnList.append(column)
            elif fc.get(FEATURE_USED) == FEATURE_USED_LENGTH:
                self._lengthColumnList.append(column)
            elif fc.get(FEATURE_USED) == FEATURE_ITEM_REAL:
                self._itemRealColumList.append(column)
            elif fc.get(FEATURE_USED) == FEATURE_USED_ATT:
                self._hardAttColumnList.append(column)
            elif fc.get(FEATURE_USED) == FEATURE_USED_DEEP_ID:
                self._itemIdColumnList.append(column)
            elif fc.get(FEATURE_USED) == FEATURE_USED_GATE:
                self._gate_judge.append(column)
            elif fc.get(FEATURE_USED) == FEATURE_USED_GROUP:
                self._groupColumnList.append(column)

        if column and fc.get(TRANSFORM_NAME) != self._transformFunctions["seq_embedding"]:
            if isinstance(column,list):
                feature_key = self._getFeatureKey(fc.get(INPUT_NAME))
                self._feature_dict[feature_key + '_id'] = [column[0]]
                self._feature_dict[feature_key + '_value'] = [column[1]]
            else:
                feature_key = self._getFeatureKey(fc.get(INPUT_NAME))
                self._feature_dict[feature_key] = [column]

        return

    def getWideColumns(self):
        return self._wideColumnList

    def getDeepColumns(self):
        return self._deepColumnList

    def getUbbEmbeddingID(self):
        return self._ubb_embedding_id

    def getUbbEmbeddingValue(self):
        return self._ubb_embedding_value

    def getFutureColumnDict(self):
        return self._feature_dict
    
    def getItemIdColumns(self):
        return self._itemIdColumnList