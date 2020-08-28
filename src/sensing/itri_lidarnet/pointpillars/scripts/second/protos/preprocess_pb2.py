# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/preprocess.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='second/protos/preprocess.proto',
  package='second.protos',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1esecond/protos/preprocess.proto\x12\rsecond.protos\"\xb1\x02\n\nPreprocess\x12&\n\x1erandom_global_rotation_min_rad\x18\x01 \x01(\x02\x12&\n\x1erandom_global_rotation_max_rad\x18\x02 \x01(\x02\x12!\n\x19random_global_scaling_min\x18\x03 \x01(\x02\x12!\n\x19random_global_scaling_max\x18\x04 \x01(\x02\x12,\n$random_noise_per_groundtruth_min_rad\x18\x05 \x01(\x02\x12,\n$random_noise_per_groundtruth_max_rad\x18\x06 \x01(\x02\x12\x31\n)random_noise_per_groundtruth_position_std\x18\x07 \x01(\x02\"\xd6\x01\n\x19\x44\x61tabasePreprocessingStep\x12\x43\n\x14\x66ilter_by_difficulty\x18\x01 \x01(\x0b\x32#.second.protos.DBFilterByDifficultyH\x00\x12U\n\x18\x66ilter_by_min_num_points\x18\x02 \x01(\x0b\x32\x31.second.protos.DBFilterByMinNumPointInGroundTruthH\x00\x42\x1d\n\x1b\x64\x61tabase_preprocessing_step\"4\n\x14\x44\x42\x46ilterByDifficulty\x12\x1c\n\x14removed_difficulties\x18\x01 \x03(\x05\"\xc3\x01\n\"DBFilterByMinNumPointInGroundTruth\x12\x64\n\x13min_num_point_pairs\x18\x01 \x03(\x0b\x32G.second.protos.DBFilterByMinNumPointInGroundTruth.MinNumPointPairsEntry\x1a\x37\n\x15MinNumPointPairsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\r:\x02\x38\x01\"\xb5\x01\n\x11PreprocessingStep\x12\x43\n\x15random_global_scaling\x18\x01 \x01(\x0b\x32\".second.protos.RandomGlobalScalingH\x00\x12\x45\n\x16random_global_rotation\x18\x02 \x01(\x0b\x32#.second.protos.RandomGlobalRotationH\x00\x42\x14\n\x12preprocessing_step\";\n\x13RandomGlobalScaling\x12\x11\n\tmin_scale\x18\x01 \x01(\x02\x12\x11\n\tmax_scale\x18\x02 \x01(\x02\"8\n\x14RandomGlobalRotation\x12\x0f\n\x07min_rad\x18\x01 \x01(\x02\x12\x0f\n\x07max_rad\x18\x02 \x01(\x02\x62\x06proto3')
)




_PREPROCESS = _descriptor.Descriptor(
  name='Preprocess',
  full_name='second.protos.Preprocess',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='random_global_rotation_min_rad', full_name='second.protos.Preprocess.random_global_rotation_min_rad', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_global_rotation_max_rad', full_name='second.protos.Preprocess.random_global_rotation_max_rad', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_global_scaling_min', full_name='second.protos.Preprocess.random_global_scaling_min', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_global_scaling_max', full_name='second.protos.Preprocess.random_global_scaling_max', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_noise_per_groundtruth_min_rad', full_name='second.protos.Preprocess.random_noise_per_groundtruth_min_rad', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_noise_per_groundtruth_max_rad', full_name='second.protos.Preprocess.random_noise_per_groundtruth_max_rad', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_noise_per_groundtruth_position_std', full_name='second.protos.Preprocess.random_noise_per_groundtruth_position_std', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=50,
  serialized_end=355,
)


_DATABASEPREPROCESSINGSTEP = _descriptor.Descriptor(
  name='DatabasePreprocessingStep',
  full_name='second.protos.DatabasePreprocessingStep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='filter_by_difficulty', full_name='second.protos.DatabasePreprocessingStep.filter_by_difficulty', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='filter_by_min_num_points', full_name='second.protos.DatabasePreprocessingStep.filter_by_min_num_points', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='database_preprocessing_step', full_name='second.protos.DatabasePreprocessingStep.database_preprocessing_step',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=358,
  serialized_end=572,
)


_DBFILTERBYDIFFICULTY = _descriptor.Descriptor(
  name='DBFilterByDifficulty',
  full_name='second.protos.DBFilterByDifficulty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='removed_difficulties', full_name='second.protos.DBFilterByDifficulty.removed_difficulties', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=574,
  serialized_end=626,
)


_DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY = _descriptor.Descriptor(
  name='MinNumPointPairsEntry',
  full_name='second.protos.DBFilterByMinNumPointInGroundTruth.MinNumPointPairsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='second.protos.DBFilterByMinNumPointInGroundTruth.MinNumPointPairsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='second.protos.DBFilterByMinNumPointInGroundTruth.MinNumPointPairsEntry.value', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=769,
  serialized_end=824,
)

_DBFILTERBYMINNUMPOINTINGROUNDTRUTH = _descriptor.Descriptor(
  name='DBFilterByMinNumPointInGroundTruth',
  full_name='second.protos.DBFilterByMinNumPointInGroundTruth',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_num_point_pairs', full_name='second.protos.DBFilterByMinNumPointInGroundTruth.min_num_point_pairs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=629,
  serialized_end=824,
)


_PREPROCESSINGSTEP = _descriptor.Descriptor(
  name='PreprocessingStep',
  full_name='second.protos.PreprocessingStep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='random_global_scaling', full_name='second.protos.PreprocessingStep.random_global_scaling', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='random_global_rotation', full_name='second.protos.PreprocessingStep.random_global_rotation', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='preprocessing_step', full_name='second.protos.PreprocessingStep.preprocessing_step',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=827,
  serialized_end=1008,
)


_RANDOMGLOBALSCALING = _descriptor.Descriptor(
  name='RandomGlobalScaling',
  full_name='second.protos.RandomGlobalScaling',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_scale', full_name='second.protos.RandomGlobalScaling.min_scale', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_scale', full_name='second.protos.RandomGlobalScaling.max_scale', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1010,
  serialized_end=1069,
)


_RANDOMGLOBALROTATION = _descriptor.Descriptor(
  name='RandomGlobalRotation',
  full_name='second.protos.RandomGlobalRotation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_rad', full_name='second.protos.RandomGlobalRotation.min_rad', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_rad', full_name='second.protos.RandomGlobalRotation.max_rad', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1071,
  serialized_end=1127,
)

_DATABASEPREPROCESSINGSTEP.fields_by_name['filter_by_difficulty'].message_type = _DBFILTERBYDIFFICULTY
_DATABASEPREPROCESSINGSTEP.fields_by_name['filter_by_min_num_points'].message_type = _DBFILTERBYMINNUMPOINTINGROUNDTRUTH
_DATABASEPREPROCESSINGSTEP.oneofs_by_name['database_preprocessing_step'].fields.append(
  _DATABASEPREPROCESSINGSTEP.fields_by_name['filter_by_difficulty'])
_DATABASEPREPROCESSINGSTEP.fields_by_name['filter_by_difficulty'].containing_oneof = _DATABASEPREPROCESSINGSTEP.oneofs_by_name['database_preprocessing_step']
_DATABASEPREPROCESSINGSTEP.oneofs_by_name['database_preprocessing_step'].fields.append(
  _DATABASEPREPROCESSINGSTEP.fields_by_name['filter_by_min_num_points'])
_DATABASEPREPROCESSINGSTEP.fields_by_name['filter_by_min_num_points'].containing_oneof = _DATABASEPREPROCESSINGSTEP.oneofs_by_name['database_preprocessing_step']
_DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY.containing_type = _DBFILTERBYMINNUMPOINTINGROUNDTRUTH
_DBFILTERBYMINNUMPOINTINGROUNDTRUTH.fields_by_name['min_num_point_pairs'].message_type = _DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY
_PREPROCESSINGSTEP.fields_by_name['random_global_scaling'].message_type = _RANDOMGLOBALSCALING
_PREPROCESSINGSTEP.fields_by_name['random_global_rotation'].message_type = _RANDOMGLOBALROTATION
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['random_global_scaling'])
_PREPROCESSINGSTEP.fields_by_name['random_global_scaling'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
_PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step'].fields.append(
  _PREPROCESSINGSTEP.fields_by_name['random_global_rotation'])
_PREPROCESSINGSTEP.fields_by_name['random_global_rotation'].containing_oneof = _PREPROCESSINGSTEP.oneofs_by_name['preprocessing_step']
DESCRIPTOR.message_types_by_name['Preprocess'] = _PREPROCESS
DESCRIPTOR.message_types_by_name['DatabasePreprocessingStep'] = _DATABASEPREPROCESSINGSTEP
DESCRIPTOR.message_types_by_name['DBFilterByDifficulty'] = _DBFILTERBYDIFFICULTY
DESCRIPTOR.message_types_by_name['DBFilterByMinNumPointInGroundTruth'] = _DBFILTERBYMINNUMPOINTINGROUNDTRUTH
DESCRIPTOR.message_types_by_name['PreprocessingStep'] = _PREPROCESSINGSTEP
DESCRIPTOR.message_types_by_name['RandomGlobalScaling'] = _RANDOMGLOBALSCALING
DESCRIPTOR.message_types_by_name['RandomGlobalRotation'] = _RANDOMGLOBALROTATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Preprocess = _reflection.GeneratedProtocolMessageType('Preprocess', (_message.Message,), dict(
  DESCRIPTOR = _PREPROCESS,
  __module__ = 'second.protos.preprocess_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.Preprocess)
  ))
_sym_db.RegisterMessage(Preprocess)

DatabasePreprocessingStep = _reflection.GeneratedProtocolMessageType('DatabasePreprocessingStep', (_message.Message,), dict(
  DESCRIPTOR = _DATABASEPREPROCESSINGSTEP,
  __module__ = 'second.protos.preprocess_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.DatabasePreprocessingStep)
  ))
_sym_db.RegisterMessage(DatabasePreprocessingStep)

DBFilterByDifficulty = _reflection.GeneratedProtocolMessageType('DBFilterByDifficulty', (_message.Message,), dict(
  DESCRIPTOR = _DBFILTERBYDIFFICULTY,
  __module__ = 'second.protos.preprocess_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.DBFilterByDifficulty)
  ))
_sym_db.RegisterMessage(DBFilterByDifficulty)

DBFilterByMinNumPointInGroundTruth = _reflection.GeneratedProtocolMessageType('DBFilterByMinNumPointInGroundTruth', (_message.Message,), dict(

  MinNumPointPairsEntry = _reflection.GeneratedProtocolMessageType('MinNumPointPairsEntry', (_message.Message,), dict(
    DESCRIPTOR = _DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY,
    __module__ = 'second.protos.preprocess_pb2'
    # @@protoc_insertion_point(class_scope:second.protos.DBFilterByMinNumPointInGroundTruth.MinNumPointPairsEntry)
    ))
  ,
  DESCRIPTOR = _DBFILTERBYMINNUMPOINTINGROUNDTRUTH,
  __module__ = 'second.protos.preprocess_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.DBFilterByMinNumPointInGroundTruth)
  ))
_sym_db.RegisterMessage(DBFilterByMinNumPointInGroundTruth)
_sym_db.RegisterMessage(DBFilterByMinNumPointInGroundTruth.MinNumPointPairsEntry)

PreprocessingStep = _reflection.GeneratedProtocolMessageType('PreprocessingStep', (_message.Message,), dict(
  DESCRIPTOR = _PREPROCESSINGSTEP,
  __module__ = 'second.protos.preprocess_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.PreprocessingStep)
  ))
_sym_db.RegisterMessage(PreprocessingStep)

RandomGlobalScaling = _reflection.GeneratedProtocolMessageType('RandomGlobalScaling', (_message.Message,), dict(
  DESCRIPTOR = _RANDOMGLOBALSCALING,
  __module__ = 'second.protos.preprocess_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.RandomGlobalScaling)
  ))
_sym_db.RegisterMessage(RandomGlobalScaling)

RandomGlobalRotation = _reflection.GeneratedProtocolMessageType('RandomGlobalRotation', (_message.Message,), dict(
  DESCRIPTOR = _RANDOMGLOBALROTATION,
  __module__ = 'second.protos.preprocess_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.RandomGlobalRotation)
  ))
_sym_db.RegisterMessage(RandomGlobalRotation)


_DBFILTERBYMINNUMPOINTINGROUNDTRUTH_MINNUMPOINTPAIRSENTRY._options = None
# @@protoc_insertion_point(module_scope)
