# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: caffe2/proto/prof_dag.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='caffe2/proto/prof_dag.proto',
  package='caffe2',
  syntax='proto2',
  serialized_pb=_b('\n\x1b\x63\x61\x66\x66\x65\x32/proto/prof_dag.proto\x12\x06\x63\x61\x66\x66\x65\x32\":\n\x0cProfDAGProto\x12\x0c\n\x04name\x18\x01 \x02(\t\x12\x0c\n\x04mean\x18\x02 \x02(\x02\x12\x0e\n\x06stddev\x18\x03 \x02(\x02\"4\n\rProfDAGProtos\x12#\n\x05stats\x18\x01 \x03(\x0b\x32\x14.caffe2.ProfDAGProto')
)




_PROFDAGPROTO = _descriptor.Descriptor(
  name='ProfDAGProto',
  full_name='caffe2.ProfDAGProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='caffe2.ProfDAGProto.name', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mean', full_name='caffe2.ProfDAGProto.mean', index=1,
      number=2, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stddev', full_name='caffe2.ProfDAGProto.stddev', index=2,
      number=3, type=2, cpp_type=6, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=39,
  serialized_end=97,
)


_PROFDAGPROTOS = _descriptor.Descriptor(
  name='ProfDAGProtos',
  full_name='caffe2.ProfDAGProtos',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='stats', full_name='caffe2.ProfDAGProtos.stats', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=99,
  serialized_end=151,
)

_PROFDAGPROTOS.fields_by_name['stats'].message_type = _PROFDAGPROTO
DESCRIPTOR.message_types_by_name['ProfDAGProto'] = _PROFDAGPROTO
DESCRIPTOR.message_types_by_name['ProfDAGProtos'] = _PROFDAGPROTOS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ProfDAGProto = _reflection.GeneratedProtocolMessageType('ProfDAGProto', (_message.Message,), dict(
  DESCRIPTOR = _PROFDAGPROTO,
  __module__ = 'caffe2.proto.prof_dag_pb2'
  # @@protoc_insertion_point(class_scope:caffe2.ProfDAGProto)
  ))
_sym_db.RegisterMessage(ProfDAGProto)

ProfDAGProtos = _reflection.GeneratedProtocolMessageType('ProfDAGProtos', (_message.Message,), dict(
  DESCRIPTOR = _PROFDAGPROTOS,
  __module__ = 'caffe2.proto.prof_dag_pb2'
  # @@protoc_insertion_point(class_scope:caffe2.ProfDAGProtos)
  ))
_sym_db.RegisterMessage(ProfDAGProtos)


# @@protoc_insertion_point(module_scope)
