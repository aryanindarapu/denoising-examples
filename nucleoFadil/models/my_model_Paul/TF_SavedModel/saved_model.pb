аф
р6«6
,
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
s
	AssignAdd
ref"TА

value"T

output_ref"TА" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
м
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Т
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

С
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
.
Identity

input"T
output"T"	
Ttype
О
ImageSummary
tag
tensor"T
summary"

max_imagesint(0"
Ttype0:
2"'
	bad_colortensorB:€  €
N
IsVariableInitialized
ref"dtypeА
is_initialized
"
dtypetypeШ
,
Log
x"T
y"T"
Ttype:

2
‘
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
о
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	Р
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
;
Minimum
x"T
y"T
z"T"
Ttype:

2	Р
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
Р
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
x
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2		"
align_cornersbool( 
p
ResizeNearestNeighborGrad

grads"T
size
output"T"
Ttype:

2"
align_cornersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
0
Round
x"T
y"T"
Ttype:

2	
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
/
Sign
x"T
y"T"
Ttype:

2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
@
Softplus
features"T
activations"T"
Ttype:
2
R
SoftplusGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
З
StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
f
TopKV2

input"T
k
values"T
indices"
sortedbool("
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И"serve*1.12.02v1.12.0-0-ga6d8ffae09еы
Ь
inputPlaceholder*
dtype0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*6
shape-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€

&down_level_0_no_0/random_uniform/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
i
$down_level_0_no_0/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *nІЃљ
i
$down_level_0_no_0/random_uniform/maxConst*
valueB
 *nІЃ=*
dtype0*
_output_shapes
: 
ƒ
.down_level_0_no_0/random_uniform/RandomUniformRandomUniform&down_level_0_no_0/random_uniform/shape*
seed±€е)*
T0*
dtype0*&
_output_shapes
: *
seed2Џў“
Ш
$down_level_0_no_0/random_uniform/subSub$down_level_0_no_0/random_uniform/max$down_level_0_no_0/random_uniform/min*
T0*
_output_shapes
: 
≤
$down_level_0_no_0/random_uniform/mulMul.down_level_0_no_0/random_uniform/RandomUniform$down_level_0_no_0/random_uniform/sub*
T0*&
_output_shapes
: 
§
 down_level_0_no_0/random_uniformAdd$down_level_0_no_0/random_uniform/mul$down_level_0_no_0/random_uniform/min*
T0*&
_output_shapes
: 
Ь
down_level_0_no_0/kernel
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 
м
down_level_0_no_0/kernel/AssignAssigndown_level_0_no_0/kernel down_level_0_no_0/random_uniform*
use_locking(*
T0*+
_class!
loc:@down_level_0_no_0/kernel*
validate_shape(*&
_output_shapes
: 
°
down_level_0_no_0/kernel/readIdentitydown_level_0_no_0/kernel*
T0*+
_class!
loc:@down_level_0_no_0/kernel*&
_output_shapes
: 
d
down_level_0_no_0/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
В
down_level_0_no_0/bias
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
—
down_level_0_no_0/bias/AssignAssigndown_level_0_no_0/biasdown_level_0_no_0/Const*
use_locking(*
T0*)
_class
loc:@down_level_0_no_0/bias*
validate_shape(*
_output_shapes
: 
П
down_level_0_no_0/bias/readIdentitydown_level_0_no_0/bias*
T0*)
_class
loc:@down_level_0_no_0/bias*
_output_shapes
: 
|
+down_level_0_no_0/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
З
down_level_0_no_0/convolutionConv2Dinputdown_level_0_no_0/kernel/read*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
√
down_level_0_no_0/BiasAddBiasAdddown_level_0_no_0/convolutiondown_level_0_no_0/bias/read*
data_formatNHWC*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
T0
Е
down_level_0_no_0/ReluReludown_level_0_no_0/BiasAdd*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 

&down_level_0_no_1/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"              
i
$down_level_0_no_1/random_uniform/minConst*
valueB
 *з”zљ*
dtype0*
_output_shapes
: 
i
$down_level_0_no_1/random_uniform/maxConst*
valueB
 *з”z=*
dtype0*
_output_shapes
: 
ƒ
.down_level_0_no_1/random_uniform/RandomUniformRandomUniform&down_level_0_no_1/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:  *
seed2юЯЩ*
seed±€е)
Ш
$down_level_0_no_1/random_uniform/subSub$down_level_0_no_1/random_uniform/max$down_level_0_no_1/random_uniform/min*
T0*
_output_shapes
: 
≤
$down_level_0_no_1/random_uniform/mulMul.down_level_0_no_1/random_uniform/RandomUniform$down_level_0_no_1/random_uniform/sub*
T0*&
_output_shapes
:  
§
 down_level_0_no_1/random_uniformAdd$down_level_0_no_1/random_uniform/mul$down_level_0_no_1/random_uniform/min*
T0*&
_output_shapes
:  
Ь
down_level_0_no_1/kernel
VariableV2*
dtype0*&
_output_shapes
:  *
	container *
shape:  *
shared_name 
м
down_level_0_no_1/kernel/AssignAssigndown_level_0_no_1/kernel down_level_0_no_1/random_uniform*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*+
_class!
loc:@down_level_0_no_1/kernel
°
down_level_0_no_1/kernel/readIdentitydown_level_0_no_1/kernel*&
_output_shapes
:  *
T0*+
_class!
loc:@down_level_0_no_1/kernel
d
down_level_0_no_1/ConstConst*
dtype0*
_output_shapes
: *
valueB *    
В
down_level_0_no_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
—
down_level_0_no_1/bias/AssignAssigndown_level_0_no_1/biasdown_level_0_no_1/Const*
use_locking(*
T0*)
_class
loc:@down_level_0_no_1/bias*
validate_shape(*
_output_shapes
: 
П
down_level_0_no_1/bias/readIdentitydown_level_0_no_1/bias*
T0*)
_class
loc:@down_level_0_no_1/bias*
_output_shapes
: 
|
+down_level_0_no_1/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Ш
down_level_0_no_1/convolutionConv2Ddown_level_0_no_0/Reludown_level_0_no_1/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
√
down_level_0_no_1/BiasAddBiasAdddown_level_0_no_1/convolutiondown_level_0_no_1/bias/read*
T0*
data_formatNHWC*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Е
down_level_0_no_1/ReluReludown_level_0_no_1/BiasAdd*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
ѕ
max_0/MaxPoolMaxPooldown_level_0_no_1/Relu*
ksize
*
paddingVALID*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
T0*
data_formatNHWC*
strides


&down_level_1_no_0/random_uniform/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
i
$down_level_1_no_0/random_uniform/minConst*
valueB
 *ЌћLљ*
dtype0*
_output_shapes
: 
i
$down_level_1_no_0/random_uniform/maxConst*
valueB
 *ЌћL=*
dtype0*
_output_shapes
: 
√
.down_level_1_no_0/random_uniform/RandomUniformRandomUniform&down_level_1_no_0/random_uniform/shape*
T0*
dtype0*&
_output_shapes
: @*
seed2»€*
seed±€е)
Ш
$down_level_1_no_0/random_uniform/subSub$down_level_1_no_0/random_uniform/max$down_level_1_no_0/random_uniform/min*
T0*
_output_shapes
: 
≤
$down_level_1_no_0/random_uniform/mulMul.down_level_1_no_0/random_uniform/RandomUniform$down_level_1_no_0/random_uniform/sub*&
_output_shapes
: @*
T0
§
 down_level_1_no_0/random_uniformAdd$down_level_1_no_0/random_uniform/mul$down_level_1_no_0/random_uniform/min*&
_output_shapes
: @*
T0
Ь
down_level_1_no_0/kernel
VariableV2*
shape: @*
shared_name *
dtype0*&
_output_shapes
: @*
	container 
м
down_level_1_no_0/kernel/AssignAssigndown_level_1_no_0/kernel down_level_1_no_0/random_uniform*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*+
_class!
loc:@down_level_1_no_0/kernel
°
down_level_1_no_0/kernel/readIdentitydown_level_1_no_0/kernel*&
_output_shapes
: @*
T0*+
_class!
loc:@down_level_1_no_0/kernel
d
down_level_1_no_0/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
В
down_level_1_no_0/bias
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
—
down_level_1_no_0/bias/AssignAssigndown_level_1_no_0/biasdown_level_1_no_0/Const*
use_locking(*
T0*)
_class
loc:@down_level_1_no_0/bias*
validate_shape(*
_output_shapes
:@
П
down_level_1_no_0/bias/readIdentitydown_level_1_no_0/bias*
T0*)
_class
loc:@down_level_1_no_0/bias*
_output_shapes
:@
|
+down_level_1_no_0/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
П
down_level_1_no_0/convolutionConv2Dmax_0/MaxPooldown_level_1_no_0/kernel/read*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
√
down_level_1_no_0/BiasAddBiasAdddown_level_1_no_0/convolutiondown_level_1_no_0/bias/read*
data_formatNHWC*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
T0
Е
down_level_1_no_0/ReluReludown_level_1_no_0/BiasAdd*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@

&down_level_1_no_1/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
i
$down_level_1_no_1/random_uniform/minConst*
valueB
 *ђ\1љ*
dtype0*
_output_shapes
: 
i
$down_level_1_no_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ђ\1=
ƒ
.down_level_1_no_1/random_uniform/RandomUniformRandomUniform&down_level_1_no_1/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:@@*
seed2дрР*
seed±€е)
Ш
$down_level_1_no_1/random_uniform/subSub$down_level_1_no_1/random_uniform/max$down_level_1_no_1/random_uniform/min*
T0*
_output_shapes
: 
≤
$down_level_1_no_1/random_uniform/mulMul.down_level_1_no_1/random_uniform/RandomUniform$down_level_1_no_1/random_uniform/sub*&
_output_shapes
:@@*
T0
§
 down_level_1_no_1/random_uniformAdd$down_level_1_no_1/random_uniform/mul$down_level_1_no_1/random_uniform/min*&
_output_shapes
:@@*
T0
Ь
down_level_1_no_1/kernel
VariableV2*
dtype0*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name 
м
down_level_1_no_1/kernel/AssignAssigndown_level_1_no_1/kernel down_level_1_no_1/random_uniform*
use_locking(*
T0*+
_class!
loc:@down_level_1_no_1/kernel*
validate_shape(*&
_output_shapes
:@@
°
down_level_1_no_1/kernel/readIdentitydown_level_1_no_1/kernel*
T0*+
_class!
loc:@down_level_1_no_1/kernel*&
_output_shapes
:@@
d
down_level_1_no_1/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
В
down_level_1_no_1/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
—
down_level_1_no_1/bias/AssignAssigndown_level_1_no_1/biasdown_level_1_no_1/Const*
use_locking(*
T0*)
_class
loc:@down_level_1_no_1/bias*
validate_shape(*
_output_shapes
:@
П
down_level_1_no_1/bias/readIdentitydown_level_1_no_1/bias*
_output_shapes
:@*
T0*)
_class
loc:@down_level_1_no_1/bias
|
+down_level_1_no_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Ш
down_level_1_no_1/convolutionConv2Ddown_level_1_no_0/Reludown_level_1_no_1/kernel/read*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
√
down_level_1_no_1/BiasAddBiasAdddown_level_1_no_1/convolutiondown_level_1_no_1/bias/read*
T0*
data_formatNHWC*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Е
down_level_1_no_1/ReluReludown_level_1_no_1/BiasAdd*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
ѕ
max_1/MaxPoolMaxPooldown_level_1_no_1/Relu*
ksize
*
paddingVALID*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
T0*
strides
*
data_formatNHWC
v
middle_0/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      @   А   
`
middle_0/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *√–љ
`
middle_0/random_uniform/maxConst*
valueB
 *√–=*
dtype0*
_output_shapes
: 
≥
%middle_0/random_uniform/RandomUniformRandomUniformmiddle_0/random_uniform/shape*
dtype0*'
_output_shapes
:@А*
seed2ђнЋ*
seed±€е)*
T0
}
middle_0/random_uniform/subSubmiddle_0/random_uniform/maxmiddle_0/random_uniform/min*
T0*
_output_shapes
: 
Ш
middle_0/random_uniform/mulMul%middle_0/random_uniform/RandomUniformmiddle_0/random_uniform/sub*
T0*'
_output_shapes
:@А
К
middle_0/random_uniformAddmiddle_0/random_uniform/mulmiddle_0/random_uniform/min*
T0*'
_output_shapes
:@А
Х
middle_0/kernel
VariableV2*
shared_name *
dtype0*'
_output_shapes
:@А*
	container *
shape:@А
…
middle_0/kernel/AssignAssignmiddle_0/kernelmiddle_0/random_uniform*
validate_shape(*'
_output_shapes
:@А*
use_locking(*
T0*"
_class
loc:@middle_0/kernel
З
middle_0/kernel/readIdentitymiddle_0/kernel*
T0*"
_class
loc:@middle_0/kernel*'
_output_shapes
:@А
]
middle_0/ConstConst*
valueBА*    *
dtype0*
_output_shapes	
:А
{
middle_0/bias
VariableV2*
shared_name *
dtype0*
_output_shapes	
:А*
	container *
shape:А
Ѓ
middle_0/bias/AssignAssignmiddle_0/biasmiddle_0/Const*
T0* 
_class
loc:@middle_0/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
u
middle_0/bias/readIdentitymiddle_0/bias*
T0* 
_class
loc:@middle_0/bias*
_output_shapes	
:А
s
"middle_0/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ю
middle_0/convolutionConv2Dmax_1/MaxPoolmiddle_0/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
	dilations
*
T0
©
middle_0/BiasAddBiasAddmiddle_0/convolutionmiddle_0/bias/read*
T0*
data_formatNHWC*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
t
middle_0/ReluRelumiddle_0/BiasAdd*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
v
middle_2/random_uniform/shapeConst*%
valueB"      А   @   *
dtype0*
_output_shapes
:
`
middle_2/random_uniform/minConst*
valueB
 *√–љ*
dtype0*
_output_shapes
: 
`
middle_2/random_uniform/maxConst*
valueB
 *√–=*
dtype0*
_output_shapes
: 
≥
%middle_2/random_uniform/RandomUniformRandomUniformmiddle_2/random_uniform/shape*
T0*
dtype0*'
_output_shapes
:А@*
seed2÷їЃ*
seed±€е)
}
middle_2/random_uniform/subSubmiddle_2/random_uniform/maxmiddle_2/random_uniform/min*
T0*
_output_shapes
: 
Ш
middle_2/random_uniform/mulMul%middle_2/random_uniform/RandomUniformmiddle_2/random_uniform/sub*
T0*'
_output_shapes
:А@
К
middle_2/random_uniformAddmiddle_2/random_uniform/mulmiddle_2/random_uniform/min*
T0*'
_output_shapes
:А@
Х
middle_2/kernel
VariableV2*
dtype0*'
_output_shapes
:А@*
	container *
shape:А@*
shared_name 
…
middle_2/kernel/AssignAssignmiddle_2/kernelmiddle_2/random_uniform*
use_locking(*
T0*"
_class
loc:@middle_2/kernel*
validate_shape(*'
_output_shapes
:А@
З
middle_2/kernel/readIdentitymiddle_2/kernel*
T0*"
_class
loc:@middle_2/kernel*'
_output_shapes
:А@
[
middle_2/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
y
middle_2/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
≠
middle_2/bias/AssignAssignmiddle_2/biasmiddle_2/Const*
T0* 
_class
loc:@middle_2/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
t
middle_2/bias/readIdentitymiddle_2/bias*
T0* 
_class
loc:@middle_2/bias*
_output_shapes
:@
s
"middle_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
э
middle_2/convolutionConv2Dmiddle_0/Relumiddle_2/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
®
middle_2/BiasAddBiasAddmiddle_2/convolutionmiddle_2/bias/read*
T0*
data_formatNHWC*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
s
middle_2/ReluRelumiddle_2/BiasAdd*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
b
up_sampling2d_1/ShapeShapemiddle_2/Relu*
T0*
out_type0*
_output_shapes
:
m
#up_sampling2d_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Ќ
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape#up_sampling2d_1/strided_slice/stack%up_sampling2d_1/strided_slice/stack_1%up_sampling2d_1/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0
f
up_sampling2d_1/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_1/mulMulup_sampling2d_1/strided_sliceup_sampling2d_1/Const*
T0*
_output_shapes
:
√
%up_sampling2d_1/ResizeNearestNeighborResizeNearestNeighbormiddle_2/Reluup_sampling2d_1/mul*
align_corners( *
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
[
concatenate_1/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
№
concatenate_1/concatConcatV2%up_sampling2d_1/ResizeNearestNeighbordown_level_1_no_1/Reluconcatenate_1/concat/axis*
N*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*

Tidx0*
T0
}
$up_level_1_no_0/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      А   @   
g
"up_level_1_no_0/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *√–љ
g
"up_level_1_no_0/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *√–=
ј
,up_level_1_no_0/random_uniform/RandomUniformRandomUniform$up_level_1_no_0/random_uniform/shape*
dtype0*'
_output_shapes
:А@*
seed2иєB*
seed±€е)*
T0
Т
"up_level_1_no_0/random_uniform/subSub"up_level_1_no_0/random_uniform/max"up_level_1_no_0/random_uniform/min*
_output_shapes
: *
T0
≠
"up_level_1_no_0/random_uniform/mulMul,up_level_1_no_0/random_uniform/RandomUniform"up_level_1_no_0/random_uniform/sub*
T0*'
_output_shapes
:А@
Я
up_level_1_no_0/random_uniformAdd"up_level_1_no_0/random_uniform/mul"up_level_1_no_0/random_uniform/min*'
_output_shapes
:А@*
T0
Ь
up_level_1_no_0/kernel
VariableV2*
shared_name *
dtype0*'
_output_shapes
:А@*
	container *
shape:А@
е
up_level_1_no_0/kernel/AssignAssignup_level_1_no_0/kernelup_level_1_no_0/random_uniform*
T0*)
_class
loc:@up_level_1_no_0/kernel*
validate_shape(*'
_output_shapes
:А@*
use_locking(
Ь
up_level_1_no_0/kernel/readIdentityup_level_1_no_0/kernel*
T0*)
_class
loc:@up_level_1_no_0/kernel*'
_output_shapes
:А@
b
up_level_1_no_0/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
А
up_level_1_no_0/bias
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
…
up_level_1_no_0/bias/AssignAssignup_level_1_no_0/biasup_level_1_no_0/Const*
use_locking(*
T0*'
_class
loc:@up_level_1_no_0/bias*
validate_shape(*
_output_shapes
:@
Й
up_level_1_no_0/bias/readIdentityup_level_1_no_0/bias*
T0*'
_class
loc:@up_level_1_no_0/bias*
_output_shapes
:@
z
)up_level_1_no_0/convolution/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
Т
up_level_1_no_0/convolutionConv2Dconcatenate_1/concatup_level_1_no_0/kernel/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
љ
up_level_1_no_0/BiasAddBiasAddup_level_1_no_0/convolutionup_level_1_no_0/bias/read*
T0*
data_formatNHWC*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Б
up_level_1_no_0/ReluReluup_level_1_no_0/BiasAdd*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
}
$up_level_1_no_2/random_uniform/shapeConst*%
valueB"      @       *
dtype0*
_output_shapes
:
g
"up_level_1_no_2/random_uniform/minConst*
valueB
 *ЌћLљ*
dtype0*
_output_shapes
: 
g
"up_level_1_no_2/random_uniform/maxConst*
valueB
 *ЌћL=*
dtype0*
_output_shapes
: 
ј
,up_level_1_no_2/random_uniform/RandomUniformRandomUniform$up_level_1_no_2/random_uniform/shape*
seed±€е)*
T0*
dtype0*&
_output_shapes
:@ *
seed2кµю
Т
"up_level_1_no_2/random_uniform/subSub"up_level_1_no_2/random_uniform/max"up_level_1_no_2/random_uniform/min*
T0*
_output_shapes
: 
ђ
"up_level_1_no_2/random_uniform/mulMul,up_level_1_no_2/random_uniform/RandomUniform"up_level_1_no_2/random_uniform/sub*&
_output_shapes
:@ *
T0
Ю
up_level_1_no_2/random_uniformAdd"up_level_1_no_2/random_uniform/mul"up_level_1_no_2/random_uniform/min*
T0*&
_output_shapes
:@ 
Ъ
up_level_1_no_2/kernel
VariableV2*
dtype0*&
_output_shapes
:@ *
	container *
shape:@ *
shared_name 
д
up_level_1_no_2/kernel/AssignAssignup_level_1_no_2/kernelup_level_1_no_2/random_uniform*
use_locking(*
T0*)
_class
loc:@up_level_1_no_2/kernel*
validate_shape(*&
_output_shapes
:@ 
Ы
up_level_1_no_2/kernel/readIdentityup_level_1_no_2/kernel*
T0*)
_class
loc:@up_level_1_no_2/kernel*&
_output_shapes
:@ 
b
up_level_1_no_2/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
А
up_level_1_no_2/bias
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
…
up_level_1_no_2/bias/AssignAssignup_level_1_no_2/biasup_level_1_no_2/Const*
use_locking(*
T0*'
_class
loc:@up_level_1_no_2/bias*
validate_shape(*
_output_shapes
: 
Й
up_level_1_no_2/bias/readIdentityup_level_1_no_2/bias*
T0*'
_class
loc:@up_level_1_no_2/bias*
_output_shapes
: 
z
)up_level_1_no_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Т
up_level_1_no_2/convolutionConv2Dup_level_1_no_0/Reluup_level_1_no_2/kernel/read*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
љ
up_level_1_no_2/BiasAddBiasAddup_level_1_no_2/convolutionup_level_1_no_2/bias/read*
T0*
data_formatNHWC*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Б
up_level_1_no_2/ReluReluup_level_1_no_2/BiasAdd*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
T0
i
up_sampling2d_2/ShapeShapeup_level_1_no_2/Relu*
_output_shapes
:*
T0*
out_type0
m
#up_sampling2d_2/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
o
%up_sampling2d_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
o
%up_sampling2d_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape#up_sampling2d_2/strided_slice/stack%up_sampling2d_2/strided_slice/stack_1%up_sampling2d_2/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
f
up_sampling2d_2/ConstConst*
valueB"      *
dtype0*
_output_shapes
:
u
up_sampling2d_2/mulMulup_sampling2d_2/strided_sliceup_sampling2d_2/Const*
T0*
_output_shapes
:
 
%up_sampling2d_2/ResizeNearestNeighborResizeNearestNeighborup_level_1_no_2/Reluup_sampling2d_2/mul*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
align_corners( *
T0
[
concatenate_2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
џ
concatenate_2/concatConcatV2%up_sampling2d_2/ResizeNearestNeighbordown_level_0_no_1/Reluconcatenate_2/concat/axis*
T0*
N*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*

Tidx0
}
$up_level_0_no_0/random_uniform/shapeConst*%
valueB"      @       *
dtype0*
_output_shapes
:
g
"up_level_0_no_0/random_uniform/minConst*
valueB
 *ЌћLљ*
dtype0*
_output_shapes
: 
g
"up_level_0_no_0/random_uniform/maxConst*
valueB
 *ЌћL=*
dtype0*
_output_shapes
: 
ј
,up_level_0_no_0/random_uniform/RandomUniformRandomUniform$up_level_0_no_0/random_uniform/shape*
dtype0*&
_output_shapes
:@ *
seed2ё√И*
seed±€е)*
T0
Т
"up_level_0_no_0/random_uniform/subSub"up_level_0_no_0/random_uniform/max"up_level_0_no_0/random_uniform/min*
T0*
_output_shapes
: 
ђ
"up_level_0_no_0/random_uniform/mulMul,up_level_0_no_0/random_uniform/RandomUniform"up_level_0_no_0/random_uniform/sub*
T0*&
_output_shapes
:@ 
Ю
up_level_0_no_0/random_uniformAdd"up_level_0_no_0/random_uniform/mul"up_level_0_no_0/random_uniform/min*
T0*&
_output_shapes
:@ 
Ъ
up_level_0_no_0/kernel
VariableV2*
dtype0*&
_output_shapes
:@ *
	container *
shape:@ *
shared_name 
д
up_level_0_no_0/kernel/AssignAssignup_level_0_no_0/kernelup_level_0_no_0/random_uniform*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*)
_class
loc:@up_level_0_no_0/kernel
Ы
up_level_0_no_0/kernel/readIdentityup_level_0_no_0/kernel*
T0*)
_class
loc:@up_level_0_no_0/kernel*&
_output_shapes
:@ 
b
up_level_0_no_0/ConstConst*
dtype0*
_output_shapes
: *
valueB *    
А
up_level_0_no_0/bias
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
…
up_level_0_no_0/bias/AssignAssignup_level_0_no_0/biasup_level_0_no_0/Const*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@up_level_0_no_0/bias
Й
up_level_0_no_0/bias/readIdentityup_level_0_no_0/bias*
_output_shapes
: *
T0*'
_class
loc:@up_level_0_no_0/bias
z
)up_level_0_no_0/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Т
up_level_0_no_0/convolutionConv2Dconcatenate_2/concatup_level_0_no_0/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
љ
up_level_0_no_0/BiasAddBiasAddup_level_0_no_0/convolutionup_level_0_no_0/bias/read*
data_formatNHWC*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
T0
Б
up_level_0_no_0/ReluReluup_level_0_no_0/BiasAdd*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
}
$up_level_0_no_2/random_uniform/shapeConst*%
valueB"              *
dtype0*
_output_shapes
:
g
"up_level_0_no_2/random_uniform/minConst*
valueB
 *з”zљ*
dtype0*
_output_shapes
: 
g
"up_level_0_no_2/random_uniform/maxConst*
valueB
 *з”z=*
dtype0*
_output_shapes
: 
ј
,up_level_0_no_2/random_uniform/RandomUniformRandomUniform$up_level_0_no_2/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:  *
seed2ШВа*
seed±€е)
Т
"up_level_0_no_2/random_uniform/subSub"up_level_0_no_2/random_uniform/max"up_level_0_no_2/random_uniform/min*
T0*
_output_shapes
: 
ђ
"up_level_0_no_2/random_uniform/mulMul,up_level_0_no_2/random_uniform/RandomUniform"up_level_0_no_2/random_uniform/sub*&
_output_shapes
:  *
T0
Ю
up_level_0_no_2/random_uniformAdd"up_level_0_no_2/random_uniform/mul"up_level_0_no_2/random_uniform/min*
T0*&
_output_shapes
:  
Ъ
up_level_0_no_2/kernel
VariableV2*
shape:  *
shared_name *
dtype0*&
_output_shapes
:  *
	container 
д
up_level_0_no_2/kernel/AssignAssignup_level_0_no_2/kernelup_level_0_no_2/random_uniform*
use_locking(*
T0*)
_class
loc:@up_level_0_no_2/kernel*
validate_shape(*&
_output_shapes
:  
Ы
up_level_0_no_2/kernel/readIdentityup_level_0_no_2/kernel*&
_output_shapes
:  *
T0*)
_class
loc:@up_level_0_no_2/kernel
b
up_level_0_no_2/ConstConst*
valueB *    *
dtype0*
_output_shapes
: 
А
up_level_0_no_2/bias
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
…
up_level_0_no_2/bias/AssignAssignup_level_0_no_2/biasup_level_0_no_2/Const*
use_locking(*
T0*'
_class
loc:@up_level_0_no_2/bias*
validate_shape(*
_output_shapes
: 
Й
up_level_0_no_2/bias/readIdentityup_level_0_no_2/bias*
_output_shapes
: *
T0*'
_class
loc:@up_level_0_no_2/bias
z
)up_level_0_no_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Т
up_level_0_no_2/convolutionConv2Dup_level_0_no_0/Reluup_level_0_no_2/kernel/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
	dilations

љ
up_level_0_no_2/BiasAddBiasAddup_level_0_no_2/convolutionup_level_0_no_2/bias/read*
T0*
data_formatNHWC*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Б
up_level_0_no_2/ReluReluup_level_0_no_2/BiasAdd*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
T0
v
conv2d_1/random_uniform/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
`
conv2d_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *JQЏЊ
`
conv2d_1/random_uniform/maxConst*
valueB
 *JQЏ>*
dtype0*
_output_shapes
: 
≤
%conv2d_1/random_uniform/RandomUniformRandomUniformconv2d_1/random_uniform/shape*
seed±€е)*
T0*
dtype0*&
_output_shapes
: *
seed2ЩЭЬ
}
conv2d_1/random_uniform/subSubconv2d_1/random_uniform/maxconv2d_1/random_uniform/min*
T0*
_output_shapes
: 
Ч
conv2d_1/random_uniform/mulMul%conv2d_1/random_uniform/RandomUniformconv2d_1/random_uniform/sub*
T0*&
_output_shapes
: 
Й
conv2d_1/random_uniformAddconv2d_1/random_uniform/mulconv2d_1/random_uniform/min*&
_output_shapes
: *
T0
У
conv2d_1/kernel
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 
»
conv2d_1/kernel/AssignAssignconv2d_1/kernelconv2d_1/random_uniform*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel
Ж
conv2d_1/kernel/readIdentityconv2d_1/kernel*&
_output_shapes
: *
T0*"
_class
loc:@conv2d_1/kernel
[
conv2d_1/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
y
conv2d_1/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
≠
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/Const*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0* 
_class
loc:@conv2d_1/bias*
_output_shapes
:
s
"conv2d_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Е
conv2d_1/convolutionConv2Dup_level_0_no_2/Reluconv2d_1/kernel/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
	dilations
*
T0
®
conv2d_1/BiasAddBiasAddconv2d_1/convolutionconv2d_1/bias/read*
data_formatNHWC*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0
u
	add_1/addAddconv2d_1/BiasAddinput*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
x
activation_1/IdentityIdentity	add_1/add*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
v
conv2d_2/random_uniform/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
`
conv2d_2/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *JQЏЊ
`
conv2d_2/random_uniform/maxConst*
valueB
 *JQЏ>*
dtype0*
_output_shapes
: 
±
%conv2d_2/random_uniform/RandomUniformRandomUniformconv2d_2/random_uniform/shape*
T0*
dtype0*&
_output_shapes
: *
seed2мзt*
seed±€е)
}
conv2d_2/random_uniform/subSubconv2d_2/random_uniform/maxconv2d_2/random_uniform/min*
T0*
_output_shapes
: 
Ч
conv2d_2/random_uniform/mulMul%conv2d_2/random_uniform/RandomUniformconv2d_2/random_uniform/sub*
T0*&
_output_shapes
: 
Й
conv2d_2/random_uniformAddconv2d_2/random_uniform/mulconv2d_2/random_uniform/min*&
_output_shapes
: *
T0
У
conv2d_2/kernel
VariableV2*
dtype0*&
_output_shapes
: *
	container *
shape: *
shared_name 
»
conv2d_2/kernel/AssignAssignconv2d_2/kernelconv2d_2/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
: 
Ж
conv2d_2/kernel/readIdentityconv2d_2/kernel*&
_output_shapes
: *
T0*"
_class
loc:@conv2d_2/kernel
[
conv2d_2/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
y
conv2d_2/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
≠
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias
t
conv2d_2/bias/readIdentityconv2d_2/bias*
T0* 
_class
loc:@conv2d_2/bias*
_output_shapes
:
s
"conv2d_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Е
conv2d_2/convolutionConv2Dup_level_0_no_2/Reluconv2d_2/kernel/read*
paddingVALID*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
®
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_2/bias/read*
T0*
data_formatNHWC*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
{
conv2d_2/SoftplusSoftplusconv2d_2/BiasAdd*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
S
lambda_1/add/yConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
В
lambda_1/addAddconv2d_2/Softpluslambda_1/add/y*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0
Ђ
lambda_1/PlaceholderPlaceholder*6
shape-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
dtype0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
U
lambda_1/add_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *oГ:
Й
lambda_1/add_1Addlambda_1/Placeholderlambda_1/add_1/y*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
[
concatenate_3/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ѕ
concatenate_3/concatConcatV2activation_1/Identitylambda_1/addconcatenate_3/concat/axis*

Tidx0*
T0*
N*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
_
Adam/iterations/initial_valueConst*
dtype0	*
_output_shapes
: *
value	B	 R 
s
Adam/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
Њ
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
use_locking(*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: 
v
Adam/iterations/readIdentityAdam/iterations*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
Z
Adam/lr/initial_valueConst*
valueB
 *Ј—9*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ю
Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
use_locking(*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: 
^
Adam/lr/readIdentityAdam/lr*
_output_shapes
: *
T0*
_class
loc:@Adam/lr
^
Adam/beta_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
o
Adam/beta_1
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
Ѓ
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
T0*
_class
loc:@Adam/beta_1*
validate_shape(*
_output_shapes
: *
use_locking(
j
Adam/beta_1/readIdentityAdam/beta_1*
_output_shapes
: *
T0*
_class
loc:@Adam/beta_1
^
Adam/beta_2/initial_valueConst*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Ѓ
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
use_locking(*
T0*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: 
j
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
™
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/decay
g
Adam/decay/readIdentity
Adam/decay*
_output_shapes
: *
T0*
_class
loc:@Adam/decay
љ
concatenate_3_targetPlaceholder*?
shape6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
dtype0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
w
concatenate_3_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
q
loss/concatenate_3_loss/ShapeShapeconcatenate_3_target*
T0*
out_type0*
_output_shapes
:
~
+loss/concatenate_3_loss/strided_slice/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-loss/concatenate_3_loss/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-loss/concatenate_3_loss/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%loss/concatenate_3_loss/strided_sliceStridedSliceloss/concatenate_3_loss/Shape+loss/concatenate_3_loss/strided_slice/stack-loss/concatenate_3_loss/strided_slice/stack_1-loss/concatenate_3_loss/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
~
-loss/concatenate_3_loss/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB"        
s
1loss/concatenate_3_loss/strided_slice_1/stack_1/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ћ
/loss/concatenate_3_loss/strided_slice_1/stack_1Pack1loss/concatenate_3_loss/strided_slice_1/stack_1/0%loss/concatenate_3_loss/strided_slice*
T0*

axis *
N*
_output_shapes
:
А
/loss/concatenate_3_loss/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
§
'loss/concatenate_3_loss/strided_slice_1StridedSliceconcatenate_3/concat-loss/concatenate_3_loss/strided_slice_1/stack/loss/concatenate_3_loss/strided_slice_1/stack_1/loss/concatenate_3_loss/strided_slice_1/stack_2*
shrink_axis_mask *
ellipsis_mask*

begin_mask*
new_axis_mask *
end_mask *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*
Index0
q
/loss/concatenate_3_loss/strided_slice_2/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
«
-loss/concatenate_3_loss/strided_slice_2/stackPack/loss/concatenate_3_loss/strided_slice_2/stack/0%loss/concatenate_3_loss/strided_slice*
T0*

axis *
N*
_output_shapes
:
А
/loss/concatenate_3_loss/strided_slice_2/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
А
/loss/concatenate_3_loss/strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
§
'loss/concatenate_3_loss/strided_slice_2StridedSliceconcatenate_3/concat-loss/concatenate_3_loss/strided_slice_2/stack/loss/concatenate_3_loss/strided_slice_2/stack_1/loss/concatenate_3_loss/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask*
new_axis_mask *
end_mask*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ґ
loss/concatenate_3_loss/subSub'loss/concatenate_3_loss/strided_slice_1concatenate_3_target*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0
≈
loss/concatenate_3_loss/truedivRealDivloss/concatenate_3_loss/sub'loss/concatenate_3_loss/strided_slice_2*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ш
loss/concatenate_3_loss/AbsAbsloss/concatenate_3_loss/truediv*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
†
loss/concatenate_3_loss/LogLog'loss/concatenate_3_loss/strided_slice_2*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
±
loss/concatenate_3_loss/addAddloss/concatenate_3_loss/Absloss/concatenate_3_loss/Log*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
d
loss/concatenate_3_loss/add_1/yConst*
valueB
 *r1?*
dtype0*
_output_shapes
: 
Ј
loss/concatenate_3_loss/add_1Addloss/concatenate_3_loss/addloss/concatenate_3_loss/add_1/y*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
y
.loss/concatenate_3_loss/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ў
loss/concatenate_3_loss/MeanMeanloss/concatenate_3_loss/add_1.loss/concatenate_3_loss/Mean/reduction_indices*
	keep_dims( *

Tidx0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Б
0loss/concatenate_3_loss/Mean_1/reduction_indicesConst*
valueB"      *
dtype0*
_output_shapes
:
Ѕ
loss/concatenate_3_loss/Mean_1Meanloss/concatenate_3_loss/Mean0loss/concatenate_3_loss/Mean_1/reduction_indices*
T0*#
_output_shapes
:€€€€€€€€€*
	keep_dims( *

Tidx0
О
loss/concatenate_3_loss/mulMulloss/concatenate_3_loss/Mean_1concatenate_3_sample_weights*
T0*#
_output_shapes
:€€€€€€€€€
g
"loss/concatenate_3_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ь
 loss/concatenate_3_loss/NotEqualNotEqualconcatenate_3_sample_weights"loss/concatenate_3_loss/NotEqual/y*
T0*#
_output_shapes
:€€€€€€€€€
У
loss/concatenate_3_loss/CastCast loss/concatenate_3_loss/NotEqual*

SrcT0
*
Truncate( *#
_output_shapes
:€€€€€€€€€*

DstT0
g
loss/concatenate_3_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
°
loss/concatenate_3_loss/Mean_2Meanloss/concatenate_3_loss/Castloss/concatenate_3_loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Ч
!loss/concatenate_3_loss/truediv_1RealDivloss/concatenate_3_loss/mulloss/concatenate_3_loss/Mean_2*
T0*#
_output_shapes
:€€€€€€€€€
i
loss/concatenate_3_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
®
loss/concatenate_3_loss/Mean_3Mean!loss/concatenate_3_loss/truediv_1loss/concatenate_3_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
\
loss/mulMul
loss/mul/xloss/concatenate_3_loss/Mean_3*
T0*
_output_shapes
: 
e
metrics/mse/ShapeShapeconcatenate_3_target*
_output_shapes
:*
T0*
out_type0
r
metrics/mse/strided_slice/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
k
!metrics/mse/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
k
!metrics/mse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
µ
metrics/mse/strided_sliceStridedSlicemetrics/mse/Shapemetrics/mse/strided_slice/stack!metrics/mse/strided_slice/stack_1!metrics/mse/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
r
!metrics/mse/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
g
%metrics/mse/strided_slice_1/stack_1/0Const*
value	B : *
dtype0*
_output_shapes
: 
І
#metrics/mse/strided_slice_1/stack_1Pack%metrics/mse/strided_slice_1/stack_1/0metrics/mse/strided_slice*
T0*

axis *
N*
_output_shapes
:
t
#metrics/mse/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ф
metrics/mse/strided_slice_1StridedSliceconcatenate_3/concat!metrics/mse/strided_slice_1/stack#metrics/mse/strided_slice_1/stack_1#metrics/mse/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask*
new_axis_mask *
end_mask *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ю
metrics/mse/subSubmetrics/mse/strided_slice_1concatenate_3_target*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0
В
metrics/mse/SquareSquaremetrics/mse/sub*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
m
"metrics/mse/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
µ
metrics/mse/MeanMeanmetrics/mse/Square"metrics/mse/Mean/reduction_indices*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
	keep_dims( *

Tidx0
f
metrics/mse/ConstConst*!
valueB"          *
dtype0*
_output_shapes
:
}
metrics/mse/Mean_1Meanmetrics/mse/Meanmetrics/mse/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
metrics/mae/ShapeShapeconcatenate_3_target*
T0*
out_type0*
_output_shapes
:
r
metrics/mae/strided_slice/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
k
!metrics/mae/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
k
!metrics/mae/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
µ
metrics/mae/strided_sliceStridedSlicemetrics/mae/Shapemetrics/mae/strided_slice/stack!metrics/mae/strided_slice/stack_1!metrics/mae/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
r
!metrics/mae/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
g
%metrics/mae/strided_slice_1/stack_1/0Const*
value	B : *
dtype0*
_output_shapes
: 
І
#metrics/mae/strided_slice_1/stack_1Pack%metrics/mae/strided_slice_1/stack_1/0metrics/mae/strided_slice*
T0*

axis *
N*
_output_shapes
:
t
#metrics/mae/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ф
metrics/mae/strided_slice_1StridedSliceconcatenate_3/concat!metrics/mae/strided_slice_1/stack#metrics/mae/strided_slice_1/stack_1#metrics/mae/strided_slice_1/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask*
new_axis_mask *
end_mask *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*
Index0
Ю
metrics/mae/subSubmetrics/mae/strided_slice_1concatenate_3_target*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0
|
metrics/mae/AbsAbsmetrics/mae/sub*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0
m
"metrics/mae/Mean/reduction_indicesConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
≤
metrics/mae/MeanMeanmetrics/mae/Abs"metrics/mae/Mean/reduction_indices*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
	keep_dims( *

Tidx0*
T0
f
metrics/mae/ConstConst*!
valueB"          *
dtype0*
_output_shapes
:
}
metrics/mae/Mean_1Meanmetrics/mae/Meanmetrics/mae/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
}
training/Adam/gradients/ShapeConst*
_class
loc:@loss/mul*
valueB *
dtype0*
_output_shapes
: 
Г
!training/Adam/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
_class
loc:@loss/mul*
valueB
 *  А?
ґ
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*
_class
loc:@loss/mul*

index_type0*
_output_shapes
: 
ђ
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/concatenate_3_loss/Mean_3*
_output_shapes
: *
T0*
_class
loc:@loss/mul
Ъ
+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
∆
Itraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Reshape/shapeConst*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*
valueB:*
dtype0*
_output_shapes
:
ђ
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Itraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Reshape/shape*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*
Tshape0*
_output_shapes
:
’
Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/ShapeShape!loss/concatenate_3_loss/truediv_1*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*
out_type0*
_output_shapes
:
√
@training/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/TileTileCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/ReshapeAtraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Shape*#
_output_shapes
:€€€€€€€€€*

Tmultiples0*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3
„
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Shape_1Shape!loss/concatenate_3_loss/truediv_1*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*
out_type0*
_output_shapes
:
є
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Shape_2Const*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*
valueB *
dtype0*
_output_shapes
: 
Њ
Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/ConstConst*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*
valueB: *
dtype0*
_output_shapes
:
Ѕ
@training/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/ProdProdCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Shape_1Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3
ј
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Const_1Const*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*
valueB: *
dtype0*
_output_shapes
:
≈
Btraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Prod_1ProdCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Shape_2Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Const_1*
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*
_output_shapes
: 
Ї
Etraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Maximum/yConst*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*
value	B :*
dtype0*
_output_shapes
: 
≠
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/MaximumMaximumBtraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Prod_1Etraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Maximum/y*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*
_output_shapes
: 
Ђ
Dtraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/floordivFloorDiv@training/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/ProdCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Maximum*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*
_output_shapes
: 
Б
@training/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/CastCastDtraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/floordiv*

SrcT0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*
Truncate( *
_output_shapes
: *

DstT0
≥
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/truedivRealDiv@training/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Tile@training/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/Cast*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_3*#
_output_shapes
:€€€€€€€€€
’
Dtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/ShapeShapeloss/concatenate_3_loss/mul*
T0*4
_class*
(&loc:@loss/concatenate_3_loss/truediv_1*
out_type0*
_output_shapes
:
њ
Ftraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/Shape_1Const*4
_class*
(&loc:@loss/concatenate_3_loss/truediv_1*
valueB *
dtype0*
_output_shapes
: 
о
Ttraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsDtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/ShapeFtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*4
_class*
(&loc:@loss/concatenate_3_loss/truediv_1
Ъ
Ftraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/RealDivRealDivCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/truedivloss/concatenate_3_loss/Mean_2*
T0*4
_class*
(&loc:@loss/concatenate_3_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
Ё
Btraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/SumSumFtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/RealDivTtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*4
_class*
(&loc:@loss/concatenate_3_loss/truediv_1*
_output_shapes
:
Ќ
Ftraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/ReshapeReshapeBtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/SumDtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*4
_class*
(&loc:@loss/concatenate_3_loss/truediv_1*
Tshape0
 
Btraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/NegNegloss/concatenate_3_loss/mul*
T0*4
_class*
(&loc:@loss/concatenate_3_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
Ы
Htraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/RealDiv_1RealDivBtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/Negloss/concatenate_3_loss/Mean_2*
T0*4
_class*
(&loc:@loss/concatenate_3_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
°
Htraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/RealDiv_2RealDivHtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/RealDiv_1loss/concatenate_3_loss/Mean_2*
T0*4
_class*
(&loc:@loss/concatenate_3_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
Љ
Btraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/mulMulCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_3_grad/truedivHtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/RealDiv_2*
T0*4
_class*
(&loc:@loss/concatenate_3_loss/truediv_1*#
_output_shapes
:€€€€€€€€€
Ё
Dtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/Sum_1SumBtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/mulVtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/BroadcastGradientArgs:1*
T0*4
_class*
(&loc:@loss/concatenate_3_loss/truediv_1*
_output_shapes
:*
	keep_dims( *

Tidx0
∆
Htraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/Reshape_1ReshapeDtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/Sum_1Ftraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/Shape_1*
T0*4
_class*
(&loc:@loss/concatenate_3_loss/truediv_1*
Tshape0*
_output_shapes
: 
ћ
>training/Adam/gradients/loss/concatenate_3_loss/mul_grad/ShapeShapeloss/concatenate_3_loss/Mean_1*
T0*.
_class$
" loc:@loss/concatenate_3_loss/mul*
out_type0*
_output_shapes
:
ћ
@training/Adam/gradients/loss/concatenate_3_loss/mul_grad/Shape_1Shapeconcatenate_3_sample_weights*
T0*.
_class$
" loc:@loss/concatenate_3_loss/mul*
out_type0*
_output_shapes
:
÷
Ntraining/Adam/gradients/loss/concatenate_3_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>training/Adam/gradients/loss/concatenate_3_loss/mul_grad/Shape@training/Adam/gradients/loss/concatenate_3_loss/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0*.
_class$
" loc:@loss/concatenate_3_loss/mul
З
<training/Adam/gradients/loss/concatenate_3_loss/mul_grad/MulMulFtraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/Reshapeconcatenate_3_sample_weights*
T0*.
_class$
" loc:@loss/concatenate_3_loss/mul*#
_output_shapes
:€€€€€€€€€
Ѕ
<training/Adam/gradients/loss/concatenate_3_loss/mul_grad/SumSum<training/Adam/gradients/loss/concatenate_3_loss/mul_grad/MulNtraining/Adam/gradients/loss/concatenate_3_loss/mul_grad/BroadcastGradientArgs*
T0*.
_class$
" loc:@loss/concatenate_3_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
µ
@training/Adam/gradients/loss/concatenate_3_loss/mul_grad/ReshapeReshape<training/Adam/gradients/loss/concatenate_3_loss/mul_grad/Sum>training/Adam/gradients/loss/concatenate_3_loss/mul_grad/Shape*
T0*.
_class$
" loc:@loss/concatenate_3_loss/mul*
Tshape0*#
_output_shapes
:€€€€€€€€€
Л
>training/Adam/gradients/loss/concatenate_3_loss/mul_grad/Mul_1Mulloss/concatenate_3_loss/Mean_1Ftraining/Adam/gradients/loss/concatenate_3_loss/truediv_1_grad/Reshape*
T0*.
_class$
" loc:@loss/concatenate_3_loss/mul*#
_output_shapes
:€€€€€€€€€
«
>training/Adam/gradients/loss/concatenate_3_loss/mul_grad/Sum_1Sum>training/Adam/gradients/loss/concatenate_3_loss/mul_grad/Mul_1Ptraining/Adam/gradients/loss/concatenate_3_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/concatenate_3_loss/mul*
_output_shapes
:
ї
Btraining/Adam/gradients/loss/concatenate_3_loss/mul_grad/Reshape_1Reshape>training/Adam/gradients/loss/concatenate_3_loss/mul_grad/Sum_1@training/Adam/gradients/loss/concatenate_3_loss/mul_grad/Shape_1*
T0*.
_class$
" loc:@loss/concatenate_3_loss/mul*
Tshape0*#
_output_shapes
:€€€€€€€€€
–
Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/ShapeShapeloss/concatenate_3_loss/Mean*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
out_type0*
_output_shapes
:
µ
@training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/SizeConst*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
Т
?training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/addAdd0loss/concatenate_3_loss/Mean_1/reduction_indices@training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Size*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
_output_shapes
:
¶
?training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/modFloorMod?training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/add@training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Size*
_output_shapes
:*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1
ј
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Shape_1Const*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
valueB:*
dtype0*
_output_shapes
:
Љ
Gtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/range/startConst*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
value	B : *
dtype0*
_output_shapes
: 
Љ
Gtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/range/deltaConst*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
щ
Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/rangeRangeGtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/range/start@training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/SizeGtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/range/delta*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
_output_shapes
:*

Tidx0
ї
Ftraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Fill/valueConst*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
њ
@training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/FillFillCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Shape_1Ftraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Fill/value*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*

index_type0*
_output_shapes
:
ƒ
Itraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/DynamicStitchDynamicStitchAtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/range?training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/modAtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Shape@training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Fill*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
N*
_output_shapes
:
Ї
Etraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Maximum/yConst*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
Є
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/MaximumMaximumItraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/DynamicStitchEtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Maximum/y*
_output_shapes
:*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1
∞
Dtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/floordivFloorDivAtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/ShapeCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Maximum*
_output_shapes
:*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1
д
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/ReshapeReshape@training/Adam/gradients/loss/concatenate_3_loss/mul_grad/ReshapeItraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/DynamicStitch*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
Tshape0
а
@training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/TileTileCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/ReshapeDtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/floordiv*

Tmultiples0*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
“
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Shape_2Shapeloss/concatenate_3_loss/Mean*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
out_type0*
_output_shapes
:
‘
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Shape_3Shapeloss/concatenate_3_loss/Mean_1*
_output_shapes
:*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
out_type0
Њ
Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/ConstConst*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
Ѕ
@training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/ProdProdCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Shape_2Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Const*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
_output_shapes
: *
	keep_dims( *

Tidx0
ј
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Const_1Const*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
≈
Btraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Prod_1ProdCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Shape_3Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1
Љ
Gtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Maximum_1/yConst*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
±
Etraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Maximum_1MaximumBtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Prod_1Gtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Maximum_1/y*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
_output_shapes
: 
ѓ
Ftraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/floordiv_1FloorDiv@training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/ProdEtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Maximum_1*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
_output_shapes
: 
Г
@training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/CastCastFtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/floordiv_1*

SrcT0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*
Truncate( *
_output_shapes
: *

DstT0
Ќ
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/truedivRealDiv@training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Tile@training/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/Cast*
T0*1
_class'
%#loc:@loss/concatenate_3_loss/Mean_1*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ќ
?training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/ShapeShapeloss/concatenate_3_loss/add_1*
_output_shapes
:*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
out_type0
±
>training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/SizeConst*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
Ж
=training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/addAdd.loss/concatenate_3_loss/Mean/reduction_indices>training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Size*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
_output_shapes
: 
Ъ
=training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/modFloorMod=training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/add>training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Size*
_output_shapes
: *
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean
µ
Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Shape_1Const*
dtype0*
_output_shapes
: */
_class%
#!loc:@loss/concatenate_3_loss/Mean*
valueB 
Є
Etraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/range/startConst*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
value	B : *
dtype0*
_output_shapes
: 
Є
Etraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/range/deltaConst*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
п
?training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/rangeRangeEtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/range/start>training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/SizeEtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/range/delta*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
_output_shapes
:*

Tidx0
Ј
Dtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Fill/valueConst*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
≥
>training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/FillFillAtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Shape_1Dtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Fill/value*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*

index_type0*
_output_shapes
: 
Є
Gtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/DynamicStitchDynamicStitch?training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/range=training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/mod?training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Shape>training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Fill*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
N*
_output_shapes
:
ґ
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Maximum/yConst*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
∞
Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/MaximumMaximumGtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/DynamicStitchCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Maximum/y*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
_output_shapes
:
®
Btraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/floordivFloorDiv?training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/ShapeAtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Maximum*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
_output_shapes
:
о
Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/ReshapeReshapeCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_1_grad/truedivGtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/DynamicStitch*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
Tshape0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
е
>training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/TileTileAtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/ReshapeBtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/floordiv*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*

Tmultiples0*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean
ѕ
Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Shape_2Shapeloss/concatenate_3_loss/add_1*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
out_type0*
_output_shapes
:
ќ
Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Shape_3Shapeloss/concatenate_3_loss/Mean*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
out_type0*
_output_shapes
:
Ї
?training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/ConstConst*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
є
>training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/ProdProdAtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Shape_2?training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
_output_shapes
: 
Љ
Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Const_1Const*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
љ
@training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Prod_1ProdAtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Shape_3Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
_output_shapes
: 
Є
Etraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Maximum_1/yConst*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
©
Ctraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Maximum_1Maximum@training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Prod_1Etraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Maximum_1/y*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
_output_shapes
: 
І
Dtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/floordiv_1FloorDiv>training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/ProdCtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Maximum_1*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
_output_shapes
: 
э
>training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/CastCastDtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/floordiv_1*

SrcT0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*
Truncate( *
_output_shapes
: *

DstT0
“
Atraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/truedivRealDiv>training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Tile>training/Adam/gradients/loss/concatenate_3_loss/Mean_grad/Cast*
T0*/
_class%
#!loc:@loss/concatenate_3_loss/Mean*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ќ
@training/Adam/gradients/loss/concatenate_3_loss/add_1_grad/ShapeShapeloss/concatenate_3_loss/add*
T0*0
_class&
$"loc:@loss/concatenate_3_loss/add_1*
out_type0*
_output_shapes
:
Ј
Btraining/Adam/gradients/loss/concatenate_3_loss/add_1_grad/Shape_1Const*0
_class&
$"loc:@loss/concatenate_3_loss/add_1*
valueB *
dtype0*
_output_shapes
: 
ё
Ptraining/Adam/gradients/loss/concatenate_3_loss/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs@training/Adam/gradients/loss/concatenate_3_loss/add_1_grad/ShapeBtraining/Adam/gradients/loss/concatenate_3_loss/add_1_grad/Shape_1*
T0*0
_class&
$"loc:@loss/concatenate_3_loss/add_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ћ
>training/Adam/gradients/loss/concatenate_3_loss/add_1_grad/SumSumAtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/truedivPtraining/Adam/gradients/loss/concatenate_3_loss/add_1_grad/BroadcastGradientArgs*
T0*0
_class&
$"loc:@loss/concatenate_3_loss/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
д
Btraining/Adam/gradients/loss/concatenate_3_loss/add_1_grad/ReshapeReshape>training/Adam/gradients/loss/concatenate_3_loss/add_1_grad/Sum@training/Adam/gradients/loss/concatenate_3_loss/add_1_grad/Shape*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*0
_class&
$"loc:@loss/concatenate_3_loss/add_1*
Tshape0
–
@training/Adam/gradients/loss/concatenate_3_loss/add_1_grad/Sum_1SumAtraining/Adam/gradients/loss/concatenate_3_loss/Mean_grad/truedivRtraining/Adam/gradients/loss/concatenate_3_loss/add_1_grad/BroadcastGradientArgs:1*
T0*0
_class&
$"loc:@loss/concatenate_3_loss/add_1*
_output_shapes
:*
	keep_dims( *

Tidx0
ґ
Dtraining/Adam/gradients/loss/concatenate_3_loss/add_1_grad/Reshape_1Reshape@training/Adam/gradients/loss/concatenate_3_loss/add_1_grad/Sum_1Btraining/Adam/gradients/loss/concatenate_3_loss/add_1_grad/Shape_1*
_output_shapes
: *
T0*0
_class&
$"loc:@loss/concatenate_3_loss/add_1*
Tshape0
…
>training/Adam/gradients/loss/concatenate_3_loss/add_grad/ShapeShapeloss/concatenate_3_loss/Abs*
T0*.
_class$
" loc:@loss/concatenate_3_loss/add*
out_type0*
_output_shapes
:
Ћ
@training/Adam/gradients/loss/concatenate_3_loss/add_grad/Shape_1Shapeloss/concatenate_3_loss/Log*
T0*.
_class$
" loc:@loss/concatenate_3_loss/add*
out_type0*
_output_shapes
:
÷
Ntraining/Adam/gradients/loss/concatenate_3_loss/add_grad/BroadcastGradientArgsBroadcastGradientArgs>training/Adam/gradients/loss/concatenate_3_loss/add_grad/Shape@training/Adam/gradients/loss/concatenate_3_loss/add_grad/Shape_1*
T0*.
_class$
" loc:@loss/concatenate_3_loss/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
«
<training/Adam/gradients/loss/concatenate_3_loss/add_grad/SumSumBtraining/Adam/gradients/loss/concatenate_3_loss/add_1_grad/ReshapeNtraining/Adam/gradients/loss/concatenate_3_loss/add_grad/BroadcastGradientArgs*
T0*.
_class$
" loc:@loss/concatenate_3_loss/add*
_output_shapes
:*
	keep_dims( *

Tidx0
№
@training/Adam/gradients/loss/concatenate_3_loss/add_grad/ReshapeReshape<training/Adam/gradients/loss/concatenate_3_loss/add_grad/Sum>training/Adam/gradients/loss/concatenate_3_loss/add_grad/Shape*
T0*.
_class$
" loc:@loss/concatenate_3_loss/add*
Tshape0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ћ
>training/Adam/gradients/loss/concatenate_3_loss/add_grad/Sum_1SumBtraining/Adam/gradients/loss/concatenate_3_loss/add_1_grad/ReshapePtraining/Adam/gradients/loss/concatenate_3_loss/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/concatenate_3_loss/add
в
Btraining/Adam/gradients/loss/concatenate_3_loss/add_grad/Reshape_1Reshape>training/Adam/gradients/loss/concatenate_3_loss/add_grad/Sum_1@training/Adam/gradients/loss/concatenate_3_loss/add_grad/Shape_1*
T0*.
_class$
" loc:@loss/concatenate_3_loss/add*
Tshape0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
л
=training/Adam/gradients/loss/concatenate_3_loss/Abs_grad/SignSignloss/concatenate_3_loss/truediv*
T0*.
_class$
" loc:@loss/concatenate_3_loss/Abs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
…
<training/Adam/gradients/loss/concatenate_3_loss/Abs_grad/mulMul@training/Adam/gradients/loss/concatenate_3_loss/add_grad/Reshape=training/Adam/gradients/loss/concatenate_3_loss/Abs_grad/Sign*
T0*.
_class$
" loc:@loss/concatenate_3_loss/Abs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ƒ
Ctraining/Adam/gradients/loss/concatenate_3_loss/Log_grad/Reciprocal
Reciprocal'loss/concatenate_3_loss/strided_slice_2C^training/Adam/gradients/loss/concatenate_3_loss/add_grad/Reshape_1*
T0*.
_class$
" loc:@loss/concatenate_3_loss/Log*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
—
<training/Adam/gradients/loss/concatenate_3_loss/Log_grad/mulMulBtraining/Adam/gradients/loss/concatenate_3_loss/add_grad/Reshape_1Ctraining/Adam/gradients/loss/concatenate_3_loss/Log_grad/Reciprocal*
T0*.
_class$
" loc:@loss/concatenate_3_loss/Log*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
—
Btraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/ShapeShapeloss/concatenate_3_loss/sub*
T0*2
_class(
&$loc:@loss/concatenate_3_loss/truediv*
out_type0*
_output_shapes
:
я
Dtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/Shape_1Shape'loss/concatenate_3_loss/strided_slice_2*
_output_shapes
:*
T0*2
_class(
&$loc:@loss/concatenate_3_loss/truediv*
out_type0
ж
Rtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/ShapeDtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/Shape_1*
T0*2
_class(
&$loc:@loss/concatenate_3_loss/truediv*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
њ
Dtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/RealDivRealDiv<training/Adam/gradients/loss/concatenate_3_loss/Abs_grad/mul'loss/concatenate_3_loss/strided_slice_2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*2
_class(
&$loc:@loss/concatenate_3_loss/truediv
’
@training/Adam/gradients/loss/concatenate_3_loss/truediv_grad/SumSumDtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/RealDivRtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*2
_class(
&$loc:@loss/concatenate_3_loss/truediv*
_output_shapes
:
м
Dtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/ReshapeReshape@training/Adam/gradients/loss/concatenate_3_loss/truediv_grad/SumBtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/Shape*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*2
_class(
&$loc:@loss/concatenate_3_loss/truediv*
Tshape0
н
@training/Adam/gradients/loss/concatenate_3_loss/truediv_grad/NegNegloss/concatenate_3_loss/sub*
T0*2
_class(
&$loc:@loss/concatenate_3_loss/truediv*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≈
Ftraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/RealDiv_1RealDiv@training/Adam/gradients/loss/concatenate_3_loss/truediv_grad/Neg'loss/concatenate_3_loss/strided_slice_2*
T0*2
_class(
&$loc:@loss/concatenate_3_loss/truediv*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ћ
Ftraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/RealDiv_2RealDivFtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/RealDiv_1'loss/concatenate_3_loss/strided_slice_2*
T0*2
_class(
&$loc:@loss/concatenate_3_loss/truediv*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
÷
@training/Adam/gradients/loss/concatenate_3_loss/truediv_grad/mulMul<training/Adam/gradients/loss/concatenate_3_loss/Abs_grad/mulFtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/RealDiv_2*
T0*2
_class(
&$loc:@loss/concatenate_3_loss/truediv*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
’
Btraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/Sum_1Sum@training/Adam/gradients/loss/concatenate_3_loss/truediv_grad/mulTtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/BroadcastGradientArgs:1*
T0*2
_class(
&$loc:@loss/concatenate_3_loss/truediv*
_output_shapes
:*
	keep_dims( *

Tidx0
т
Ftraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/Reshape_1ReshapeBtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/Sum_1Dtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/Shape_1*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*2
_class(
&$loc:@loss/concatenate_3_loss/truediv*
Tshape0
’
>training/Adam/gradients/loss/concatenate_3_loss/sub_grad/ShapeShape'loss/concatenate_3_loss/strided_slice_1*
T0*.
_class$
" loc:@loss/concatenate_3_loss/sub*
out_type0*
_output_shapes
:
ƒ
@training/Adam/gradients/loss/concatenate_3_loss/sub_grad/Shape_1Shapeconcatenate_3_target*
T0*.
_class$
" loc:@loss/concatenate_3_loss/sub*
out_type0*
_output_shapes
:
÷
Ntraining/Adam/gradients/loss/concatenate_3_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs>training/Adam/gradients/loss/concatenate_3_loss/sub_grad/Shape@training/Adam/gradients/loss/concatenate_3_loss/sub_grad/Shape_1*
T0*.
_class$
" loc:@loss/concatenate_3_loss/sub*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
…
<training/Adam/gradients/loss/concatenate_3_loss/sub_grad/SumSumDtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/ReshapeNtraining/Adam/gradients/loss/concatenate_3_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/concatenate_3_loss/sub
№
@training/Adam/gradients/loss/concatenate_3_loss/sub_grad/ReshapeReshape<training/Adam/gradients/loss/concatenate_3_loss/sub_grad/Sum>training/Adam/gradients/loss/concatenate_3_loss/sub_grad/Shape*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*.
_class$
" loc:@loss/concatenate_3_loss/sub*
Tshape0
Ќ
>training/Adam/gradients/loss/concatenate_3_loss/sub_grad/Sum_1SumDtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/ReshapePtraining/Adam/gradients/loss/concatenate_3_loss/sub_grad/BroadcastGradientArgs:1*
T0*.
_class$
" loc:@loss/concatenate_3_loss/sub*
_output_shapes
:*
	keep_dims( *

Tidx0
÷
<training/Adam/gradients/loss/concatenate_3_loss/sub_grad/NegNeg>training/Adam/gradients/loss/concatenate_3_loss/sub_grad/Sum_1*
T0*.
_class$
" loc:@loss/concatenate_3_loss/sub*
_output_shapes
:
а
Btraining/Adam/gradients/loss/concatenate_3_loss/sub_grad/Reshape_1Reshape<training/Adam/gradients/loss/concatenate_3_loss/sub_grad/Neg@training/Adam/gradients/loss/concatenate_3_loss/sub_grad/Shape_1*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*.
_class$
" loc:@loss/concatenate_3_loss/sub*
Tshape0
Є
training/Adam/gradients/AddNAddN<training/Adam/gradients/loss/concatenate_3_loss/Log_grad/mulFtraining/Adam/gradients/loss/concatenate_3_loss/truediv_grad/Reshape_1*
T0*.
_class$
" loc:@loss/concatenate_3_loss/Log*
N*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Џ
Jtraining/Adam/gradients/loss/concatenate_3_loss/strided_slice_2_grad/ShapeShapeconcatenate_3/concat*
T0*:
_class0
.,loc:@loss/concatenate_3_loss/strided_slice_2*
out_type0*
_output_shapes
:
Ё
Utraining/Adam/gradients/loss/concatenate_3_loss/strided_slice_2_grad/StridedSliceGradStridedSliceGradJtraining/Adam/gradients/loss/concatenate_3_loss/strided_slice_2_grad/Shape-loss/concatenate_3_loss/strided_slice_2/stack/loss/concatenate_3_loss/strided_slice_2/stack_1/loss/concatenate_3_loss/strided_slice_2/stack_2training/Adam/gradients/AddN*
shrink_axis_mask *
ellipsis_mask*

begin_mask *
new_axis_mask *
end_mask*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Index0*
T0*:
_class0
.,loc:@loss/concatenate_3_loss/strided_slice_2
Џ
Jtraining/Adam/gradients/loss/concatenate_3_loss/strided_slice_1_grad/ShapeShapeconcatenate_3/concat*
T0*:
_class0
.,loc:@loss/concatenate_3_loss/strided_slice_1*
out_type0*
_output_shapes
:
Б
Utraining/Adam/gradients/loss/concatenate_3_loss/strided_slice_1_grad/StridedSliceGradStridedSliceGradJtraining/Adam/gradients/loss/concatenate_3_loss/strided_slice_1_grad/Shape-loss/concatenate_3_loss/strided_slice_1/stack/loss/concatenate_3_loss/strided_slice_1/stack_1/loss/concatenate_3_loss/strided_slice_1/stack_2@training/Adam/gradients/loss/concatenate_3_loss/sub_grad/Reshape*
shrink_axis_mask *
ellipsis_mask*

begin_mask*
new_axis_mask *
end_mask *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Index0*
T0*:
_class0
.,loc:@loss/concatenate_3_loss/strided_slice_1
е
training/Adam/gradients/AddN_1AddNUtraining/Adam/gradients/loss/concatenate_3_loss/strided_slice_2_grad/StridedSliceGradUtraining/Adam/gradients/loss/concatenate_3_loss/strided_slice_1_grad/StridedSliceGrad*
N*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*:
_class0
.,loc:@loss/concatenate_3_loss/strided_slice_2
°
6training/Adam/gradients/concatenate_3/concat_grad/RankConst*'
_class
loc:@concatenate_3/concat*
value	B :*
dtype0*
_output_shapes
: 
ё
5training/Adam/gradients/concatenate_3/concat_grad/modFloorModconcatenate_3/concat/axis6training/Adam/gradients/concatenate_3/concat_grad/Rank*
T0*'
_class
loc:@concatenate_3/concat*
_output_shapes
: 
µ
7training/Adam/gradients/concatenate_3/concat_grad/ShapeShapeactivation_1/Identity*
T0*'
_class
loc:@concatenate_3/concat*
out_type0*
_output_shapes
:
‘
8training/Adam/gradients/concatenate_3/concat_grad/ShapeNShapeNactivation_1/Identitylambda_1/add*
T0*'
_class
loc:@concatenate_3/concat*
out_type0*
N* 
_output_shapes
::
ѕ
>training/Adam/gradients/concatenate_3/concat_grad/ConcatOffsetConcatOffset5training/Adam/gradients/concatenate_3/concat_grad/mod8training/Adam/gradients/concatenate_3/concat_grad/ShapeN:training/Adam/gradients/concatenate_3/concat_grad/ShapeN:1*'
_class
loc:@concatenate_3/concat*
N* 
_output_shapes
::
№
7training/Adam/gradients/concatenate_3/concat_grad/SliceSlicetraining/Adam/gradients/AddN_1>training/Adam/gradients/concatenate_3/concat_grad/ConcatOffset8training/Adam/gradients/concatenate_3/concat_grad/ShapeN*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Index0*
T0*'
_class
loc:@concatenate_3/concat
в
9training/Adam/gradients/concatenate_3/concat_grad/Slice_1Slicetraining/Adam/gradients/AddN_1@training/Adam/gradients/concatenate_3/concat_grad/ConcatOffset:1:training/Adam/gradients/concatenate_3/concat_grad/ShapeN:1*
Index0*
T0*'
_class
loc:@concatenate_3/concat*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
°
/training/Adam/gradients/lambda_1/add_grad/ShapeShapeconv2d_2/Softplus*
T0*
_class
loc:@lambda_1/add*
out_type0*
_output_shapes
:
Х
1training/Adam/gradients/lambda_1/add_grad/Shape_1Const*
_class
loc:@lambda_1/add*
valueB *
dtype0*
_output_shapes
: 
Ъ
?training/Adam/gradients/lambda_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs/training/Adam/gradients/lambda_1/add_grad/Shape1training/Adam/gradients/lambda_1/add_grad/Shape_1*
T0*
_class
loc:@lambda_1/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
С
-training/Adam/gradients/lambda_1/add_grad/SumSum9training/Adam/gradients/concatenate_3/concat_grad/Slice_1?training/Adam/gradients/lambda_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_class
loc:@lambda_1/add*
_output_shapes
:
Ч
1training/Adam/gradients/lambda_1/add_grad/ReshapeReshape-training/Adam/gradients/lambda_1/add_grad/Sum/training/Adam/gradients/lambda_1/add_grad/Shape*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*
_class
loc:@lambda_1/add*
Tshape0
Х
/training/Adam/gradients/lambda_1/add_grad/Sum_1Sum9training/Adam/gradients/concatenate_3/concat_grad/Slice_1Atraining/Adam/gradients/lambda_1/add_grad/BroadcastGradientArgs:1*
T0*
_class
loc:@lambda_1/add*
_output_shapes
:*
	keep_dims( *

Tidx0
т
3training/Adam/gradients/lambda_1/add_grad/Reshape_1Reshape/training/Adam/gradients/lambda_1/add_grad/Sum_11training/Adam/gradients/lambda_1/add_grad/Shape_1*
_output_shapes
: *
T0*
_class
loc:@lambda_1/add*
Tshape0
Ъ
,training/Adam/gradients/add_1/add_grad/ShapeShapeconv2d_1/BiasAdd*
T0*
_class
loc:@add_1/add*
out_type0*
_output_shapes
:
С
.training/Adam/gradients/add_1/add_grad/Shape_1Shapeinput*
T0*
_class
loc:@add_1/add*
out_type0*
_output_shapes
:
О
<training/Adam/gradients/add_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs,training/Adam/gradients/add_1/add_grad/Shape.training/Adam/gradients/add_1/add_grad/Shape_1*
T0*
_class
loc:@add_1/add*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ж
*training/Adam/gradients/add_1/add_grad/SumSum7training/Adam/gradients/concatenate_3/concat_grad/Slice<training/Adam/gradients/add_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*
_class
loc:@add_1/add
Л
.training/Adam/gradients/add_1/add_grad/ReshapeReshape*training/Adam/gradients/add_1/add_grad/Sum,training/Adam/gradients/add_1/add_grad/Shape*
T0*
_class
loc:@add_1/add*
Tshape0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
К
,training/Adam/gradients/add_1/add_grad/Sum_1Sum7training/Adam/gradients/concatenate_3/concat_grad/Slice>training/Adam/gradients/add_1/add_grad/BroadcastGradientArgs:1*
T0*
_class
loc:@add_1/add*
_output_shapes
:*
	keep_dims( *

Tidx0
С
0training/Adam/gradients/add_1/add_grad/Reshape_1Reshape,training/Adam/gradients/add_1/add_grad/Sum_1.training/Adam/gradients/add_1/add_grad/Shape_1*
T0*
_class
loc:@add_1/add*
Tshape0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
В
;training/Adam/gradients/conv2d_2/Softplus_grad/SoftplusGradSoftplusGrad1training/Adam/gradients/lambda_1/add_grad/Reshapeconv2d_2/BiasAdd*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*$
_class
loc:@conv2d_2/Softplus
ў
9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad.training/Adam/gradients/add_1/add_grad/Reshape*
data_formatNHWC*
_output_shapes
:*
T0*#
_class
loc:@conv2d_1/BiasAdd
ж
9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad;training/Adam/gradients/conv2d_2/Softplus_grad/SoftplusGrad*
data_formatNHWC*
_output_shapes
:*
T0*#
_class
loc:@conv2d_2/BiasAdd
џ
8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNShapeNup_level_0_no_2/Reluconv2d_1/kernel/read*
T0*'
_class
loc:@conv2d_1/convolution*
out_type0*
N* 
_output_shapes
::
ј
Etraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_1/convolution_grad/ShapeNconv2d_1/kernel/read.training/Adam/gradients/add_1/add_grad/Reshape*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
©
Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterup_level_0_no_2/Relu:training/Adam/gradients/conv2d_1/convolution_grad/ShapeN:1.training/Adam/gradients/add_1/add_grad/Reshape*
paddingVALID*&
_output_shapes
: *
	dilations
*
T0*'
_class
loc:@conv2d_1/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
џ
8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNShapeNup_level_0_no_2/Reluconv2d_2/kernel/read*
N* 
_output_shapes
::*
T0*'
_class
loc:@conv2d_2/convolution*
out_type0
Ќ
Etraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/conv2d_2/convolution_grad/ShapeNconv2d_2/kernel/read;training/Adam/gradients/conv2d_2/Softplus_grad/SoftplusGrad*
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
ґ
Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterup_level_0_no_2/Relu:training/Adam/gradients/conv2d_2/convolution_grad/ShapeN:1;training/Adam/gradients/conv2d_2/Softplus_grad/SoftplusGrad*&
_output_shapes
: *
	dilations
*
T0*'
_class
loc:@conv2d_2/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
≤
training/Adam/gradients/AddN_2AddNEtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropInputEtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*
T0*'
_class
loc:@conv2d_1/convolution*
N*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
с
:training/Adam/gradients/up_level_0_no_2/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_2up_level_0_no_2/Relu*
T0*'
_class
loc:@up_level_0_no_2/Relu*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
у
@training/Adam/gradients/up_level_0_no_2/BiasAdd_grad/BiasAddGradBiasAddGrad:training/Adam/gradients/up_level_0_no_2/Relu_grad/ReluGrad*
T0**
_class 
loc:@up_level_0_no_2/BiasAdd*
data_formatNHWC*
_output_shapes
: 
р
?training/Adam/gradients/up_level_0_no_2/convolution_grad/ShapeNShapeNup_level_0_no_0/Reluup_level_0_no_2/kernel/read*
T0*.
_class$
" loc:@up_level_0_no_2/convolution*
out_type0*
N* 
_output_shapes
::
з
Ltraining/Adam/gradients/up_level_0_no_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput?training/Adam/gradients/up_level_0_no_2/convolution_grad/ShapeNup_level_0_no_2/kernel/read:training/Adam/gradients/up_level_0_no_2/Relu_grad/ReluGrad*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
	dilations
*
T0*.
_class$
" loc:@up_level_0_no_2/convolution
…
Mtraining/Adam/gradients/up_level_0_no_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterup_level_0_no_0/ReluAtraining/Adam/gradients/up_level_0_no_2/convolution_grad/ShapeN:1:training/Adam/gradients/up_level_0_no_2/Relu_grad/ReluGrad*&
_output_shapes
:  *
	dilations
*
T0*.
_class$
" loc:@up_level_0_no_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Я
:training/Adam/gradients/up_level_0_no_0/Relu_grad/ReluGradReluGradLtraining/Adam/gradients/up_level_0_no_2/convolution_grad/Conv2DBackpropInputup_level_0_no_0/Relu*
T0*'
_class
loc:@up_level_0_no_0/Relu*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
у
@training/Adam/gradients/up_level_0_no_0/BiasAdd_grad/BiasAddGradBiasAddGrad:training/Adam/gradients/up_level_0_no_0/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0**
_class 
loc:@up_level_0_no_0/BiasAdd
р
?training/Adam/gradients/up_level_0_no_0/convolution_grad/ShapeNShapeNconcatenate_2/concatup_level_0_no_0/kernel/read*
T0*.
_class$
" loc:@up_level_0_no_0/convolution*
out_type0*
N* 
_output_shapes
::
з
Ltraining/Adam/gradients/up_level_0_no_0/convolution_grad/Conv2DBackpropInputConv2DBackpropInput?training/Adam/gradients/up_level_0_no_0/convolution_grad/ShapeNup_level_0_no_0/kernel/read:training/Adam/gradients/up_level_0_no_0/Relu_grad/ReluGrad*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
	dilations
*
T0*.
_class$
" loc:@up_level_0_no_0/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
…
Mtraining/Adam/gradients/up_level_0_no_0/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconcatenate_2/concatAtraining/Adam/gradients/up_level_0_no_0/convolution_grad/ShapeN:1:training/Adam/gradients/up_level_0_no_0/Relu_grad/ReluGrad*
	dilations
*
T0*.
_class$
" loc:@up_level_0_no_0/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:@ 
°
6training/Adam/gradients/concatenate_2/concat_grad/RankConst*'
_class
loc:@concatenate_2/concat*
value	B :*
dtype0*
_output_shapes
: 
ё
5training/Adam/gradients/concatenate_2/concat_grad/modFloorModconcatenate_2/concat/axis6training/Adam/gradients/concatenate_2/concat_grad/Rank*
T0*'
_class
loc:@concatenate_2/concat*
_output_shapes
: 
≈
7training/Adam/gradients/concatenate_2/concat_grad/ShapeShape%up_sampling2d_2/ResizeNearestNeighbor*
T0*'
_class
loc:@concatenate_2/concat*
out_type0*
_output_shapes
:
о
8training/Adam/gradients/concatenate_2/concat_grad/ShapeNShapeN%up_sampling2d_2/ResizeNearestNeighbordown_level_0_no_1/Relu*
T0*'
_class
loc:@concatenate_2/concat*
out_type0*
N* 
_output_shapes
::
ѕ
>training/Adam/gradients/concatenate_2/concat_grad/ConcatOffsetConcatOffset5training/Adam/gradients/concatenate_2/concat_grad/mod8training/Adam/gradients/concatenate_2/concat_grad/ShapeN:training/Adam/gradients/concatenate_2/concat_grad/ShapeN:1*'
_class
loc:@concatenate_2/concat*
N* 
_output_shapes
::
К
7training/Adam/gradients/concatenate_2/concat_grad/SliceSliceLtraining/Adam/gradients/up_level_0_no_0/convolution_grad/Conv2DBackpropInput>training/Adam/gradients/concatenate_2/concat_grad/ConcatOffset8training/Adam/gradients/concatenate_2/concat_grad/ShapeN*
Index0*
T0*'
_class
loc:@concatenate_2/concat*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Р
9training/Adam/gradients/concatenate_2/concat_grad/Slice_1SliceLtraining/Adam/gradients/up_level_0_no_0/convolution_grad/Conv2DBackpropInput@training/Adam/gradients/concatenate_2/concat_grad/ConcatOffset:1:training/Adam/gradients/concatenate_2/concat_grad/ShapeN:1*
Index0*
T0*'
_class
loc:@concatenate_2/concat*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
÷
Htraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ShapeShapeup_level_1_no_2/Relu*
T0*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*
out_type0*
_output_shapes
:
Џ
Vtraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/strided_slice/stackConst*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*
valueB:*
dtype0*
_output_shapes
:
№
Xtraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*
valueB:
№
Xtraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*
valueB:
Ж
Ptraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/strided_sliceStridedSliceHtraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ShapeVtraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/strided_slice/stackXtraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/strided_slice/stack_1Xtraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor
Я
\training/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGrad7training/Adam/gradients/concatenate_2/concat_grad/SlicePtraining/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/strided_slice*
align_corners( *
T0*8
_class.
,*loc:@up_sampling2d_2/ResizeNearestNeighbor*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
ѓ
:training/Adam/gradients/up_level_1_no_2/Relu_grad/ReluGradReluGrad\training/Adam/gradients/up_sampling2d_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradup_level_1_no_2/Relu*
T0*'
_class
loc:@up_level_1_no_2/Relu*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
у
@training/Adam/gradients/up_level_1_no_2/BiasAdd_grad/BiasAddGradBiasAddGrad:training/Adam/gradients/up_level_1_no_2/Relu_grad/ReluGrad*
T0**
_class 
loc:@up_level_1_no_2/BiasAdd*
data_formatNHWC*
_output_shapes
: 
р
?training/Adam/gradients/up_level_1_no_2/convolution_grad/ShapeNShapeNup_level_1_no_0/Reluup_level_1_no_2/kernel/read*
T0*.
_class$
" loc:@up_level_1_no_2/convolution*
out_type0*
N* 
_output_shapes
::
з
Ltraining/Adam/gradients/up_level_1_no_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput?training/Adam/gradients/up_level_1_no_2/convolution_grad/ShapeNup_level_1_no_2/kernel/read:training/Adam/gradients/up_level_1_no_2/Relu_grad/ReluGrad*
T0*.
_class$
" loc:@up_level_1_no_2/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
	dilations

…
Mtraining/Adam/gradients/up_level_1_no_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterup_level_1_no_0/ReluAtraining/Adam/gradients/up_level_1_no_2/convolution_grad/ShapeN:1:training/Adam/gradients/up_level_1_no_2/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
:@ *
	dilations
*
T0*.
_class$
" loc:@up_level_1_no_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
Я
:training/Adam/gradients/up_level_1_no_0/Relu_grad/ReluGradReluGradLtraining/Adam/gradients/up_level_1_no_2/convolution_grad/Conv2DBackpropInputup_level_1_no_0/Relu*
T0*'
_class
loc:@up_level_1_no_0/Relu*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
у
@training/Adam/gradients/up_level_1_no_0/BiasAdd_grad/BiasAddGradBiasAddGrad:training/Adam/gradients/up_level_1_no_0/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0**
_class 
loc:@up_level_1_no_0/BiasAdd
р
?training/Adam/gradients/up_level_1_no_0/convolution_grad/ShapeNShapeNconcatenate_1/concatup_level_1_no_0/kernel/read*
T0*.
_class$
" loc:@up_level_1_no_0/convolution*
out_type0*
N* 
_output_shapes
::
и
Ltraining/Adam/gradients/up_level_1_no_0/convolution_grad/Conv2DBackpropInputConv2DBackpropInput?training/Adam/gradients/up_level_1_no_0/convolution_grad/ShapeNup_level_1_no_0/kernel/read:training/Adam/gradients/up_level_1_no_0/Relu_grad/ReluGrad*
	dilations
*
T0*.
_class$
" loc:@up_level_1_no_0/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
Mtraining/Adam/gradients/up_level_1_no_0/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterconcatenate_1/concatAtraining/Adam/gradients/up_level_1_no_0/convolution_grad/ShapeN:1:training/Adam/gradients/up_level_1_no_0/Relu_grad/ReluGrad*
T0*.
_class$
" loc:@up_level_1_no_0/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:А@*
	dilations

°
6training/Adam/gradients/concatenate_1/concat_grad/RankConst*
dtype0*
_output_shapes
: *'
_class
loc:@concatenate_1/concat*
value	B :
ё
5training/Adam/gradients/concatenate_1/concat_grad/modFloorModconcatenate_1/concat/axis6training/Adam/gradients/concatenate_1/concat_grad/Rank*
_output_shapes
: *
T0*'
_class
loc:@concatenate_1/concat
≈
7training/Adam/gradients/concatenate_1/concat_grad/ShapeShape%up_sampling2d_1/ResizeNearestNeighbor*
_output_shapes
:*
T0*'
_class
loc:@concatenate_1/concat*
out_type0
о
8training/Adam/gradients/concatenate_1/concat_grad/ShapeNShapeN%up_sampling2d_1/ResizeNearestNeighbordown_level_1_no_1/Relu*
T0*'
_class
loc:@concatenate_1/concat*
out_type0*
N* 
_output_shapes
::
ѕ
>training/Adam/gradients/concatenate_1/concat_grad/ConcatOffsetConcatOffset5training/Adam/gradients/concatenate_1/concat_grad/mod8training/Adam/gradients/concatenate_1/concat_grad/ShapeN:training/Adam/gradients/concatenate_1/concat_grad/ShapeN:1*'
_class
loc:@concatenate_1/concat*
N* 
_output_shapes
::
К
7training/Adam/gradients/concatenate_1/concat_grad/SliceSliceLtraining/Adam/gradients/up_level_1_no_0/convolution_grad/Conv2DBackpropInput>training/Adam/gradients/concatenate_1/concat_grad/ConcatOffset8training/Adam/gradients/concatenate_1/concat_grad/ShapeN*
Index0*
T0*'
_class
loc:@concatenate_1/concat*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Р
9training/Adam/gradients/concatenate_1/concat_grad/Slice_1SliceLtraining/Adam/gradients/up_level_1_no_0/convolution_grad/Conv2DBackpropInput@training/Adam/gradients/concatenate_1/concat_grad/ConcatOffset:1:training/Adam/gradients/concatenate_1/concat_grad/ShapeN:1*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
Index0*
T0*'
_class
loc:@concatenate_1/concat
ѕ
Htraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ShapeShapemiddle_2/Relu*
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
out_type0*
_output_shapes
:
Џ
Vtraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/strided_slice/stackConst*
dtype0*
_output_shapes
:*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
valueB:
№
Xtraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
valueB:
№
Xtraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/strided_slice/stack_2Const*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*
valueB:*
dtype0*
_output_shapes
:
Ж
Ptraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/strided_sliceStridedSliceHtraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ShapeVtraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/strided_slice/stackXtraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/strided_slice/stack_1Xtraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor
Я
\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGrad7training/Adam/gradients/concatenate_1/concat_grad/SlicePtraining/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/strided_slice*
T0*8
_class.
,*loc:@up_sampling2d_1/ResizeNearestNeighbor*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
align_corners( 
Ъ
3training/Adam/gradients/middle_2/Relu_grad/ReluGradReluGrad\training/Adam/gradients/up_sampling2d_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradmiddle_2/Relu*
T0* 
_class
loc:@middle_2/Relu*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
ё
9training/Adam/gradients/middle_2/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/middle_2/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0*#
_class
loc:@middle_2/BiasAdd
‘
8training/Adam/gradients/middle_2/convolution_grad/ShapeNShapeNmiddle_0/Relumiddle_2/kernel/read*
N* 
_output_shapes
::*
T0*'
_class
loc:@middle_2/convolution*
out_type0
≈
Etraining/Adam/gradients/middle_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/middle_2/convolution_grad/ShapeNmiddle_2/kernel/read3training/Adam/gradients/middle_2/Relu_grad/ReluGrad*
T0*'
_class
loc:@middle_2/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
	dilations

І
Ftraining/Adam/gradients/middle_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermiddle_0/Relu:training/Adam/gradients/middle_2/convolution_grad/ShapeN:13training/Adam/gradients/middle_2/Relu_grad/ReluGrad*'
_output_shapes
:А@*
	dilations
*
T0*'
_class
loc:@middle_2/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
Д
3training/Adam/gradients/middle_0/Relu_grad/ReluGradReluGradEtraining/Adam/gradients/middle_2/convolution_grad/Conv2DBackpropInputmiddle_0/Relu*
T0* 
_class
loc:@middle_0/Relu*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
я
9training/Adam/gradients/middle_0/BiasAdd_grad/BiasAddGradBiasAddGrad3training/Adam/gradients/middle_0/Relu_grad/ReluGrad*
T0*#
_class
loc:@middle_0/BiasAdd*
data_formatNHWC*
_output_shapes	
:А
‘
8training/Adam/gradients/middle_0/convolution_grad/ShapeNShapeNmax_1/MaxPoolmiddle_0/kernel/read*
T0*'
_class
loc:@middle_0/convolution*
out_type0*
N* 
_output_shapes
::
ƒ
Etraining/Adam/gradients/middle_0/convolution_grad/Conv2DBackpropInputConv2DBackpropInput8training/Adam/gradients/middle_0/convolution_grad/ShapeNmiddle_0/kernel/read3training/Adam/gradients/middle_0/Relu_grad/ReluGrad*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
	dilations
*
T0*'
_class
loc:@middle_0/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
І
Ftraining/Adam/gradients/middle_0/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_1/MaxPool:training/Adam/gradients/middle_0/convolution_grad/ShapeN:13training/Adam/gradients/middle_0/Relu_grad/ReluGrad*
	dilations
*
T0*'
_class
loc:@middle_0/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*'
_output_shapes
:@А
ф
6training/Adam/gradients/max_1/MaxPool_grad/MaxPoolGradMaxPoolGraddown_level_1_no_1/Relumax_1/MaxPoolEtraining/Adam/gradients/middle_0/convolution_grad/Conv2DBackpropInput*
ksize
*
paddingVALID*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
T0* 
_class
loc:@max_1/MaxPool*
strides
*
data_formatNHWC
Ч
training/Adam/gradients/AddN_3AddN9training/Adam/gradients/concatenate_1/concat_grad/Slice_16training/Adam/gradients/max_1/MaxPool_grad/MaxPoolGrad*
T0*'
_class
loc:@concatenate_1/concat*
N*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
ч
<training/Adam/gradients/down_level_1_no_1/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_3down_level_1_no_1/Relu*
T0*)
_class
loc:@down_level_1_no_1/Relu*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
щ
Btraining/Adam/gradients/down_level_1_no_1/BiasAdd_grad/BiasAddGradBiasAddGrad<training/Adam/gradients/down_level_1_no_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0*,
_class"
 loc:@down_level_1_no_1/BiasAdd
ш
Atraining/Adam/gradients/down_level_1_no_1/convolution_grad/ShapeNShapeNdown_level_1_no_0/Reludown_level_1_no_1/kernel/read*
N* 
_output_shapes
::*
T0*0
_class&
$"loc:@down_level_1_no_1/convolution*
out_type0
с
Ntraining/Adam/gradients/down_level_1_no_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInputAtraining/Adam/gradients/down_level_1_no_1/convolution_grad/ShapeNdown_level_1_no_1/kernel/read<training/Adam/gradients/down_level_1_no_1/Relu_grad/ReluGrad*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
	dilations
*
T0*0
_class&
$"loc:@down_level_1_no_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
”
Otraining/Adam/gradients/down_level_1_no_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdown_level_1_no_0/ReluCtraining/Adam/gradients/down_level_1_no_1/convolution_grad/ShapeN:1<training/Adam/gradients/down_level_1_no_1/Relu_grad/ReluGrad*
paddingSAME*&
_output_shapes
:@@*
	dilations
*
T0*0
_class&
$"loc:@down_level_1_no_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
І
<training/Adam/gradients/down_level_1_no_0/Relu_grad/ReluGradReluGradNtraining/Adam/gradients/down_level_1_no_1/convolution_grad/Conv2DBackpropInputdown_level_1_no_0/Relu*
T0*)
_class
loc:@down_level_1_no_0/Relu*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
щ
Btraining/Adam/gradients/down_level_1_no_0/BiasAdd_grad/BiasAddGradBiasAddGrad<training/Adam/gradients/down_level_1_no_0/Relu_grad/ReluGrad*
T0*,
_class"
 loc:@down_level_1_no_0/BiasAdd*
data_formatNHWC*
_output_shapes
:@
п
Atraining/Adam/gradients/down_level_1_no_0/convolution_grad/ShapeNShapeNmax_0/MaxPooldown_level_1_no_0/kernel/read*
T0*0
_class&
$"loc:@down_level_1_no_0/convolution*
out_type0*
N* 
_output_shapes
::
с
Ntraining/Adam/gradients/down_level_1_no_0/convolution_grad/Conv2DBackpropInputConv2DBackpropInputAtraining/Adam/gradients/down_level_1_no_0/convolution_grad/ShapeNdown_level_1_no_0/kernel/read<training/Adam/gradients/down_level_1_no_0/Relu_grad/ReluGrad*
	dilations
*
T0*0
_class&
$"loc:@down_level_1_no_0/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
Otraining/Adam/gradients/down_level_1_no_0/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_0/MaxPoolCtraining/Adam/gradients/down_level_1_no_0/convolution_grad/ShapeN:1<training/Adam/gradients/down_level_1_no_0/Relu_grad/ReluGrad*
	dilations
*
T0*0
_class&
$"loc:@down_level_1_no_0/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @
э
6training/Adam/gradients/max_0/MaxPool_grad/MaxPoolGradMaxPoolGraddown_level_0_no_1/Relumax_0/MaxPoolNtraining/Adam/gradients/down_level_1_no_0/convolution_grad/Conv2DBackpropInput*
T0* 
_class
loc:@max_0/MaxPool*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ч
training/Adam/gradients/AddN_4AddN9training/Adam/gradients/concatenate_2/concat_grad/Slice_16training/Adam/gradients/max_0/MaxPool_grad/MaxPoolGrad*
N*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
T0*'
_class
loc:@concatenate_2/concat
ч
<training/Adam/gradients/down_level_0_no_1/Relu_grad/ReluGradReluGradtraining/Adam/gradients/AddN_4down_level_0_no_1/Relu*
T0*)
_class
loc:@down_level_0_no_1/Relu*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
щ
Btraining/Adam/gradients/down_level_0_no_1/BiasAdd_grad/BiasAddGradBiasAddGrad<training/Adam/gradients/down_level_0_no_1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0*,
_class"
 loc:@down_level_0_no_1/BiasAdd
ш
Atraining/Adam/gradients/down_level_0_no_1/convolution_grad/ShapeNShapeNdown_level_0_no_0/Reludown_level_0_no_1/kernel/read*
T0*0
_class&
$"loc:@down_level_0_no_1/convolution*
out_type0*
N* 
_output_shapes
::
с
Ntraining/Adam/gradients/down_level_0_no_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInputAtraining/Adam/gradients/down_level_0_no_1/convolution_grad/ShapeNdown_level_0_no_1/kernel/read<training/Adam/gradients/down_level_0_no_1/Relu_grad/ReluGrad*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
	dilations
*
T0*0
_class&
$"loc:@down_level_0_no_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
”
Otraining/Adam/gradients/down_level_0_no_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterdown_level_0_no_0/ReluCtraining/Adam/gradients/down_level_0_no_1/convolution_grad/ShapeN:1<training/Adam/gradients/down_level_0_no_1/Relu_grad/ReluGrad*
T0*0
_class&
$"loc:@down_level_0_no_1/convolution*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:  *
	dilations

І
<training/Adam/gradients/down_level_0_no_0/Relu_grad/ReluGradReluGradNtraining/Adam/gradients/down_level_0_no_1/convolution_grad/Conv2DBackpropInputdown_level_0_no_0/Relu*
T0*)
_class
loc:@down_level_0_no_0/Relu*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
щ
Btraining/Adam/gradients/down_level_0_no_0/BiasAdd_grad/BiasAddGradBiasAddGrad<training/Adam/gradients/down_level_0_no_0/Relu_grad/ReluGrad*
T0*,
_class"
 loc:@down_level_0_no_0/BiasAdd*
data_formatNHWC*
_output_shapes
: 
з
Atraining/Adam/gradients/down_level_0_no_0/convolution_grad/ShapeNShapeNinputdown_level_0_no_0/kernel/read*
T0*0
_class&
$"loc:@down_level_0_no_0/convolution*
out_type0*
N* 
_output_shapes
::
с
Ntraining/Adam/gradients/down_level_0_no_0/convolution_grad/Conv2DBackpropInputConv2DBackpropInputAtraining/Adam/gradients/down_level_0_no_0/convolution_grad/ShapeNdown_level_0_no_0/kernel/read<training/Adam/gradients/down_level_0_no_0/Relu_grad/ReluGrad*
paddingSAME*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
	dilations
*
T0*0
_class&
$"loc:@down_level_0_no_0/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
¬
Otraining/Adam/gradients/down_level_0_no_0/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterinputCtraining/Adam/gradients/down_level_0_no_0/convolution_grad/ShapeN:1<training/Adam/gradients/down_level_0_no_0/Relu_grad/ReluGrad*
	dilations
*
T0*0
_class&
$"loc:@down_level_0_no_0/convolution*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: 
_
training/Adam/AssignAdd/valueConst*
dtype0	*
_output_shapes
: *
value	B	 R
ђ
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: *
use_locking( 
p
training/Adam/CastCastAdam/iterations/read*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
X
training/Adam/add/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
T0*
_output_shapes
: 
X
training/Adam/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  А
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
_output_shapes
: *
T0
Б
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
T0*
_output_shapes
: 
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
_output_shapes
: *
T0
x
training/Adam/zerosConst*%
valueB *    *
dtype0*&
_output_shapes
: 
Ъ
training/Adam/Variable
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 
ў
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/zeros*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
: 
Ы
training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*)
_class
loc:@training/Adam/Variable*&
_output_shapes
: 
b
training/Adam/zeros_1Const*
valueB *    *
dtype0*
_output_shapes
: 
Д
training/Adam/Variable_1
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
’
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/zeros_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
: *
use_locking(
Х
training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
_output_shapes
: *
T0*+
_class!
loc:@training/Adam/Variable_1
~
%training/Adam/zeros_2/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"              
`
training/Adam/zeros_2/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
training/Adam/zeros_2Fill%training/Adam/zeros_2/shape_as_tensortraining/Adam/zeros_2/Const*
T0*

index_type0*&
_output_shapes
:  
Ь
training/Adam/Variable_2
VariableV2*
dtype0*&
_output_shapes
:  *
	container *
shape:  *
shared_name 
б
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/zeros_2*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2
°
training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0*+
_class!
loc:@training/Adam/Variable_2*&
_output_shapes
:  
b
training/Adam/zeros_3Const*
dtype0*
_output_shapes
: *
valueB *    
Д
training/Adam/Variable_3
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
’
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/zeros_3*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
: 
Х
training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
_output_shapes
: *
T0*+
_class!
loc:@training/Adam/Variable_3
~
%training/Adam/zeros_4/shape_as_tensorConst*%
valueB"          @   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_4/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
training/Adam/zeros_4Fill%training/Adam/zeros_4/shape_as_tensortraining/Adam/zeros_4/Const*
T0*

index_type0*&
_output_shapes
: @
Ь
training/Adam/Variable_4
VariableV2*
shape: @*
shared_name *
dtype0*&
_output_shapes
: @*
	container 
б
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/zeros_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*&
_output_shapes
: @
°
training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4*&
_output_shapes
: @
b
training/Adam/zeros_5Const*
dtype0*
_output_shapes
:@*
valueB@*    
Д
training/Adam/Variable_5
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
’
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/zeros_5*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:@
Х
training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
_output_shapes
:@*
T0*+
_class!
loc:@training/Adam/Variable_5
~
%training/Adam/zeros_6/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_6/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
training/Adam/zeros_6Fill%training/Adam/zeros_6/shape_as_tensortraining/Adam/zeros_6/Const*
T0*

index_type0*&
_output_shapes
:@@
Ь
training/Adam/Variable_6
VariableV2*
dtype0*&
_output_shapes
:@@*
	container *
shape:@@*
shared_name 
б
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/zeros_6*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*&
_output_shapes
:@@*
use_locking(
°
training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*+
_class!
loc:@training/Adam/Variable_6*&
_output_shapes
:@@
b
training/Adam/zeros_7Const*
dtype0*
_output_shapes
:@*
valueB@*    
Д
training/Adam/Variable_7
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
’
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/zeros_7*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7
Х
training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
T0*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:@
~
%training/Adam/zeros_8/shape_as_tensorConst*%
valueB"      @   А   *
dtype0*
_output_shapes
:
`
training/Adam/zeros_8/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
•
training/Adam/zeros_8Fill%training/Adam/zeros_8/shape_as_tensortraining/Adam/zeros_8/Const*
T0*

index_type0*'
_output_shapes
:@А
Ю
training/Adam/Variable_8
VariableV2*
shared_name *
dtype0*'
_output_shapes
:@А*
	container *
shape:@А
в
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/zeros_8*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*'
_output_shapes
:@А
Ґ
training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
T0*+
_class!
loc:@training/Adam/Variable_8*'
_output_shapes
:@А
d
training/Adam/zeros_9Const*
valueBА*    *
dtype0*
_output_shapes	
:А
Ж
training/Adam/Variable_9
VariableV2*
shape:А*
shared_name *
dtype0*
_output_shapes	
:А*
	container 
÷
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/zeros_9*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes	
:А
Ц
training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes	
:А

&training/Adam/zeros_10/shape_as_tensorConst*%
valueB"      А   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_10/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
®
training/Adam/zeros_10Fill&training/Adam/zeros_10/shape_as_tensortraining/Adam/zeros_10/Const*
T0*

index_type0*'
_output_shapes
:А@
Я
training/Adam/Variable_10
VariableV2*
dtype0*'
_output_shapes
:А@*
	container *
shape:А@*
shared_name 
ж
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/zeros_10*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*'
_output_shapes
:А@
•
training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10*'
_output_shapes
:А@
c
training/Adam/zeros_11Const*
valueB@*    *
dtype0*
_output_shapes
:@
Е
training/Adam/Variable_11
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
ў
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/zeros_11*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:@
Ш
training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
:@

&training/Adam/zeros_12/shape_as_tensorConst*%
valueB"      А   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_12/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
®
training/Adam/zeros_12Fill&training/Adam/zeros_12/shape_as_tensortraining/Adam/zeros_12/Const*
T0*

index_type0*'
_output_shapes
:А@
Я
training/Adam/Variable_12
VariableV2*
dtype0*'
_output_shapes
:А@*
	container *
shape:А@*
shared_name 
ж
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/zeros_12*
validate_shape(*'
_output_shapes
:А@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12
•
training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12*'
_output_shapes
:А@
c
training/Adam/zeros_13Const*
valueB@*    *
dtype0*
_output_shapes
:@
Е
training/Adam/Variable_13
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
ў
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/zeros_13*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_13
Ш
training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_13

&training/Adam/zeros_14/shape_as_tensorConst*%
valueB"      @       *
dtype0*
_output_shapes
:
a
training/Adam/zeros_14/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
І
training/Adam/zeros_14Fill&training/Adam/zeros_14/shape_as_tensortraining/Adam/zeros_14/Const*
T0*

index_type0*&
_output_shapes
:@ 
Э
training/Adam/Variable_14
VariableV2*
shape:@ *
shared_name *
dtype0*&
_output_shapes
:@ *
	container 
е
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/zeros_14*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14
§
training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*&
_output_shapes
:@ *
T0*,
_class"
 loc:@training/Adam/Variable_14
c
training/Adam/zeros_15Const*
valueB *    *
dtype0*
_output_shapes
: 
Е
training/Adam/Variable_15
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
ў
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/zeros_15*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15
Ш
training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
: 

&training/Adam/zeros_16/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"      @       
a
training/Adam/zeros_16/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
І
training/Adam/zeros_16Fill&training/Adam/zeros_16/shape_as_tensortraining/Adam/zeros_16/Const*
T0*

index_type0*&
_output_shapes
:@ 
Э
training/Adam/Variable_16
VariableV2*
dtype0*&
_output_shapes
:@ *
	container *
shape:@ *
shared_name 
е
 training/Adam/Variable_16/AssignAssigntraining/Adam/Variable_16training/Adam/zeros_16*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16
§
training/Adam/Variable_16/readIdentitytraining/Adam/Variable_16*&
_output_shapes
:@ *
T0*,
_class"
 loc:@training/Adam/Variable_16
c
training/Adam/zeros_17Const*
valueB *    *
dtype0*
_output_shapes
: 
Е
training/Adam/Variable_17
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
ў
 training/Adam/Variable_17/AssignAssigntraining/Adam/Variable_17training/Adam/zeros_17*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
: 
Ш
training/Adam/Variable_17/readIdentitytraining/Adam/Variable_17*
_output_shapes
: *
T0*,
_class"
 loc:@training/Adam/Variable_17

&training/Adam/zeros_18/shape_as_tensorConst*%
valueB"              *
dtype0*
_output_shapes
:
a
training/Adam/zeros_18/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
І
training/Adam/zeros_18Fill&training/Adam/zeros_18/shape_as_tensortraining/Adam/zeros_18/Const*
T0*

index_type0*&
_output_shapes
:  
Э
training/Adam/Variable_18
VariableV2*
shared_name *
dtype0*&
_output_shapes
:  *
	container *
shape:  
е
 training/Adam/Variable_18/AssignAssigntraining/Adam/Variable_18training/Adam/zeros_18*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_18
§
training/Adam/Variable_18/readIdentitytraining/Adam/Variable_18*&
_output_shapes
:  *
T0*,
_class"
 loc:@training/Adam/Variable_18
c
training/Adam/zeros_19Const*
valueB *    *
dtype0*
_output_shapes
: 
Е
training/Adam/Variable_19
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
ў
 training/Adam/Variable_19/AssignAssigntraining/Adam/Variable_19training/Adam/zeros_19*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19*
validate_shape(*
_output_shapes
: 
Ш
training/Adam/Variable_19/readIdentitytraining/Adam/Variable_19*
T0*,
_class"
 loc:@training/Adam/Variable_19*
_output_shapes
: 
{
training/Adam/zeros_20Const*%
valueB *    *
dtype0*&
_output_shapes
: 
Э
training/Adam/Variable_20
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 
е
 training/Adam/Variable_20/AssignAssigntraining/Adam/Variable_20training/Adam/zeros_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*&
_output_shapes
: 
§
training/Adam/Variable_20/readIdentitytraining/Adam/Variable_20*&
_output_shapes
: *
T0*,
_class"
 loc:@training/Adam/Variable_20
c
training/Adam/zeros_21Const*
dtype0*
_output_shapes
:*
valueB*    
Е
training/Adam/Variable_21
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ў
 training/Adam/Variable_21/AssignAssigntraining/Adam/Variable_21training/Adam/zeros_21*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_21
Ш
training/Adam/Variable_21/readIdentitytraining/Adam/Variable_21*
T0*,
_class"
 loc:@training/Adam/Variable_21*
_output_shapes
:
{
training/Adam/zeros_22Const*%
valueB *    *
dtype0*&
_output_shapes
: 
Э
training/Adam/Variable_22
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 
е
 training/Adam/Variable_22/AssignAssigntraining/Adam/Variable_22training/Adam/zeros_22*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*&
_output_shapes
: *
use_locking(
§
training/Adam/Variable_22/readIdentitytraining/Adam/Variable_22*
T0*,
_class"
 loc:@training/Adam/Variable_22*&
_output_shapes
: 
c
training/Adam/zeros_23Const*
dtype0*
_output_shapes
:*
valueB*    
Е
training/Adam/Variable_23
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ў
 training/Adam/Variable_23/AssignAssigntraining/Adam/Variable_23training/Adam/zeros_23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23
Ш
training/Adam/Variable_23/readIdentitytraining/Adam/Variable_23*
T0*,
_class"
 loc:@training/Adam/Variable_23*
_output_shapes
:
{
training/Adam/zeros_24Const*%
valueB *    *
dtype0*&
_output_shapes
: 
Э
training/Adam/Variable_24
VariableV2*
dtype0*&
_output_shapes
: *
	container *
shape: *
shared_name 
е
 training/Adam/Variable_24/AssignAssigntraining/Adam/Variable_24training/Adam/zeros_24*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*&
_output_shapes
: 
§
training/Adam/Variable_24/readIdentitytraining/Adam/Variable_24*
T0*,
_class"
 loc:@training/Adam/Variable_24*&
_output_shapes
: 
c
training/Adam/zeros_25Const*
dtype0*
_output_shapes
: *
valueB *    
Е
training/Adam/Variable_25
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
ў
 training/Adam/Variable_25/AssignAssigntraining/Adam/Variable_25training/Adam/zeros_25*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
: 
Ш
training/Adam/Variable_25/readIdentitytraining/Adam/Variable_25*
T0*,
_class"
 loc:@training/Adam/Variable_25*
_output_shapes
: 

&training/Adam/zeros_26/shape_as_tensorConst*%
valueB"              *
dtype0*
_output_shapes
:
a
training/Adam/zeros_26/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
І
training/Adam/zeros_26Fill&training/Adam/zeros_26/shape_as_tensortraining/Adam/zeros_26/Const*
T0*

index_type0*&
_output_shapes
:  
Э
training/Adam/Variable_26
VariableV2*
shape:  *
shared_name *
dtype0*&
_output_shapes
:  *
	container 
е
 training/Adam/Variable_26/AssignAssigntraining/Adam/Variable_26training/Adam/zeros_26*
T0*,
_class"
 loc:@training/Adam/Variable_26*
validate_shape(*&
_output_shapes
:  *
use_locking(
§
training/Adam/Variable_26/readIdentitytraining/Adam/Variable_26*
T0*,
_class"
 loc:@training/Adam/Variable_26*&
_output_shapes
:  
c
training/Adam/zeros_27Const*
valueB *    *
dtype0*
_output_shapes
: 
Е
training/Adam/Variable_27
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
ў
 training/Adam/Variable_27/AssignAssigntraining/Adam/Variable_27training/Adam/zeros_27*
T0*,
_class"
 loc:@training/Adam/Variable_27*
validate_shape(*
_output_shapes
: *
use_locking(
Ш
training/Adam/Variable_27/readIdentitytraining/Adam/Variable_27*
T0*,
_class"
 loc:@training/Adam/Variable_27*
_output_shapes
: 

&training/Adam/zeros_28/shape_as_tensorConst*%
valueB"          @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_28/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
І
training/Adam/zeros_28Fill&training/Adam/zeros_28/shape_as_tensortraining/Adam/zeros_28/Const*
T0*

index_type0*&
_output_shapes
: @
Э
training/Adam/Variable_28
VariableV2*
shape: @*
shared_name *
dtype0*&
_output_shapes
: @*
	container 
е
 training/Adam/Variable_28/AssignAssigntraining/Adam/Variable_28training/Adam/zeros_28*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_28
§
training/Adam/Variable_28/readIdentitytraining/Adam/Variable_28*
T0*,
_class"
 loc:@training/Adam/Variable_28*&
_output_shapes
: @
c
training/Adam/zeros_29Const*
valueB@*    *
dtype0*
_output_shapes
:@
Е
training/Adam/Variable_29
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
ў
 training/Adam/Variable_29/AssignAssigntraining/Adam/Variable_29training/Adam/zeros_29*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_29*
validate_shape(*
_output_shapes
:@
Ш
training/Adam/Variable_29/readIdentitytraining/Adam/Variable_29*
T0*,
_class"
 loc:@training/Adam/Variable_29*
_output_shapes
:@

&training/Adam/zeros_30/shape_as_tensorConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_30/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
І
training/Adam/zeros_30Fill&training/Adam/zeros_30/shape_as_tensortraining/Adam/zeros_30/Const*&
_output_shapes
:@@*
T0*

index_type0
Э
training/Adam/Variable_30
VariableV2*
shared_name *
dtype0*&
_output_shapes
:@@*
	container *
shape:@@
е
 training/Adam/Variable_30/AssignAssigntraining/Adam/Variable_30training/Adam/zeros_30*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_30*
validate_shape(*&
_output_shapes
:@@
§
training/Adam/Variable_30/readIdentitytraining/Adam/Variable_30*
T0*,
_class"
 loc:@training/Adam/Variable_30*&
_output_shapes
:@@
c
training/Adam/zeros_31Const*
valueB@*    *
dtype0*
_output_shapes
:@
Е
training/Adam/Variable_31
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
ў
 training/Adam/Variable_31/AssignAssigntraining/Adam/Variable_31training/Adam/zeros_31*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_31
Ш
training/Adam/Variable_31/readIdentitytraining/Adam/Variable_31*
T0*,
_class"
 loc:@training/Adam/Variable_31*
_output_shapes
:@

&training/Adam/zeros_32/shape_as_tensorConst*%
valueB"      @   А   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_32/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
®
training/Adam/zeros_32Fill&training/Adam/zeros_32/shape_as_tensortraining/Adam/zeros_32/Const*
T0*

index_type0*'
_output_shapes
:@А
Я
training/Adam/Variable_32
VariableV2*
shape:@А*
shared_name *
dtype0*'
_output_shapes
:@А*
	container 
ж
 training/Adam/Variable_32/AssignAssigntraining/Adam/Variable_32training/Adam/zeros_32*
T0*,
_class"
 loc:@training/Adam/Variable_32*
validate_shape(*'
_output_shapes
:@А*
use_locking(
•
training/Adam/Variable_32/readIdentitytraining/Adam/Variable_32*
T0*,
_class"
 loc:@training/Adam/Variable_32*'
_output_shapes
:@А
e
training/Adam/zeros_33Const*
valueBА*    *
dtype0*
_output_shapes	
:А
З
training/Adam/Variable_33
VariableV2*
shared_name *
dtype0*
_output_shapes	
:А*
	container *
shape:А
Џ
 training/Adam/Variable_33/AssignAssigntraining/Adam/Variable_33training/Adam/zeros_33*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_33
Щ
training/Adam/Variable_33/readIdentitytraining/Adam/Variable_33*
_output_shapes	
:А*
T0*,
_class"
 loc:@training/Adam/Variable_33

&training/Adam/zeros_34/shape_as_tensorConst*%
valueB"      А   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_34/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
®
training/Adam/zeros_34Fill&training/Adam/zeros_34/shape_as_tensortraining/Adam/zeros_34/Const*'
_output_shapes
:А@*
T0*

index_type0
Я
training/Adam/Variable_34
VariableV2*
shared_name *
dtype0*'
_output_shapes
:А@*
	container *
shape:А@
ж
 training/Adam/Variable_34/AssignAssigntraining/Adam/Variable_34training/Adam/zeros_34*
validate_shape(*'
_output_shapes
:А@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_34
•
training/Adam/Variable_34/readIdentitytraining/Adam/Variable_34*'
_output_shapes
:А@*
T0*,
_class"
 loc:@training/Adam/Variable_34
c
training/Adam/zeros_35Const*
valueB@*    *
dtype0*
_output_shapes
:@
Е
training/Adam/Variable_35
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
ў
 training/Adam/Variable_35/AssignAssigntraining/Adam/Variable_35training/Adam/zeros_35*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_35*
validate_shape(*
_output_shapes
:@
Ш
training/Adam/Variable_35/readIdentitytraining/Adam/Variable_35*
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_35

&training/Adam/zeros_36/shape_as_tensorConst*%
valueB"      А   @   *
dtype0*
_output_shapes
:
a
training/Adam/zeros_36/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
®
training/Adam/zeros_36Fill&training/Adam/zeros_36/shape_as_tensortraining/Adam/zeros_36/Const*
T0*

index_type0*'
_output_shapes
:А@
Я
training/Adam/Variable_36
VariableV2*
dtype0*'
_output_shapes
:А@*
	container *
shape:А@*
shared_name 
ж
 training/Adam/Variable_36/AssignAssigntraining/Adam/Variable_36training/Adam/zeros_36*
validate_shape(*'
_output_shapes
:А@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_36
•
training/Adam/Variable_36/readIdentitytraining/Adam/Variable_36*'
_output_shapes
:А@*
T0*,
_class"
 loc:@training/Adam/Variable_36
c
training/Adam/zeros_37Const*
valueB@*    *
dtype0*
_output_shapes
:@
Е
training/Adam/Variable_37
VariableV2*
shape:@*
shared_name *
dtype0*
_output_shapes
:@*
	container 
ў
 training/Adam/Variable_37/AssignAssigntraining/Adam/Variable_37training/Adam/zeros_37*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_37*
validate_shape(*
_output_shapes
:@
Ш
training/Adam/Variable_37/readIdentitytraining/Adam/Variable_37*
_output_shapes
:@*
T0*,
_class"
 loc:@training/Adam/Variable_37

&training/Adam/zeros_38/shape_as_tensorConst*%
valueB"      @       *
dtype0*
_output_shapes
:
a
training/Adam/zeros_38/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
І
training/Adam/zeros_38Fill&training/Adam/zeros_38/shape_as_tensortraining/Adam/zeros_38/Const*
T0*

index_type0*&
_output_shapes
:@ 
Э
training/Adam/Variable_38
VariableV2*
dtype0*&
_output_shapes
:@ *
	container *
shape:@ *
shared_name 
е
 training/Adam/Variable_38/AssignAssigntraining/Adam/Variable_38training/Adam/zeros_38*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_38
§
training/Adam/Variable_38/readIdentitytraining/Adam/Variable_38*
T0*,
_class"
 loc:@training/Adam/Variable_38*&
_output_shapes
:@ 
c
training/Adam/zeros_39Const*
valueB *    *
dtype0*
_output_shapes
: 
Е
training/Adam/Variable_39
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
ў
 training/Adam/Variable_39/AssignAssigntraining/Adam/Variable_39training/Adam/zeros_39*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_39*
validate_shape(*
_output_shapes
: 
Ш
training/Adam/Variable_39/readIdentitytraining/Adam/Variable_39*
T0*,
_class"
 loc:@training/Adam/Variable_39*
_output_shapes
: 

&training/Adam/zeros_40/shape_as_tensorConst*%
valueB"      @       *
dtype0*
_output_shapes
:
a
training/Adam/zeros_40/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
І
training/Adam/zeros_40Fill&training/Adam/zeros_40/shape_as_tensortraining/Adam/zeros_40/Const*&
_output_shapes
:@ *
T0*

index_type0
Э
training/Adam/Variable_40
VariableV2*
dtype0*&
_output_shapes
:@ *
	container *
shape:@ *
shared_name 
е
 training/Adam/Variable_40/AssignAssigntraining/Adam/Variable_40training/Adam/zeros_40*
T0*,
_class"
 loc:@training/Adam/Variable_40*
validate_shape(*&
_output_shapes
:@ *
use_locking(
§
training/Adam/Variable_40/readIdentitytraining/Adam/Variable_40*
T0*,
_class"
 loc:@training/Adam/Variable_40*&
_output_shapes
:@ 
c
training/Adam/zeros_41Const*
valueB *    *
dtype0*
_output_shapes
: 
Е
training/Adam/Variable_41
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
ў
 training/Adam/Variable_41/AssignAssigntraining/Adam/Variable_41training/Adam/zeros_41*
T0*,
_class"
 loc:@training/Adam/Variable_41*
validate_shape(*
_output_shapes
: *
use_locking(
Ш
training/Adam/Variable_41/readIdentitytraining/Adam/Variable_41*
T0*,
_class"
 loc:@training/Adam/Variable_41*
_output_shapes
: 

&training/Adam/zeros_42/shape_as_tensorConst*
dtype0*
_output_shapes
:*%
valueB"              
a
training/Adam/zeros_42/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
І
training/Adam/zeros_42Fill&training/Adam/zeros_42/shape_as_tensortraining/Adam/zeros_42/Const*
T0*

index_type0*&
_output_shapes
:  
Э
training/Adam/Variable_42
VariableV2*
shape:  *
shared_name *
dtype0*&
_output_shapes
:  *
	container 
е
 training/Adam/Variable_42/AssignAssigntraining/Adam/Variable_42training/Adam/zeros_42*
T0*,
_class"
 loc:@training/Adam/Variable_42*
validate_shape(*&
_output_shapes
:  *
use_locking(
§
training/Adam/Variable_42/readIdentitytraining/Adam/Variable_42*
T0*,
_class"
 loc:@training/Adam/Variable_42*&
_output_shapes
:  
c
training/Adam/zeros_43Const*
valueB *    *
dtype0*
_output_shapes
: 
Е
training/Adam/Variable_43
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
ў
 training/Adam/Variable_43/AssignAssigntraining/Adam/Variable_43training/Adam/zeros_43*
T0*,
_class"
 loc:@training/Adam/Variable_43*
validate_shape(*
_output_shapes
: *
use_locking(
Ш
training/Adam/Variable_43/readIdentitytraining/Adam/Variable_43*
T0*,
_class"
 loc:@training/Adam/Variable_43*
_output_shapes
: 
{
training/Adam/zeros_44Const*
dtype0*&
_output_shapes
: *%
valueB *    
Э
training/Adam/Variable_44
VariableV2*
shape: *
shared_name *
dtype0*&
_output_shapes
: *
	container 
е
 training/Adam/Variable_44/AssignAssigntraining/Adam/Variable_44training/Adam/zeros_44*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_44
§
training/Adam/Variable_44/readIdentitytraining/Adam/Variable_44*
T0*,
_class"
 loc:@training/Adam/Variable_44*&
_output_shapes
: 
c
training/Adam/zeros_45Const*
valueB*    *
dtype0*
_output_shapes
:
Е
training/Adam/Variable_45
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
ў
 training/Adam/Variable_45/AssignAssigntraining/Adam/Variable_45training/Adam/zeros_45*
T0*,
_class"
 loc:@training/Adam/Variable_45*
validate_shape(*
_output_shapes
:*
use_locking(
Ш
training/Adam/Variable_45/readIdentitytraining/Adam/Variable_45*
T0*,
_class"
 loc:@training/Adam/Variable_45*
_output_shapes
:
{
training/Adam/zeros_46Const*%
valueB *    *
dtype0*&
_output_shapes
: 
Э
training/Adam/Variable_46
VariableV2*
shared_name *
dtype0*&
_output_shapes
: *
	container *
shape: 
е
 training/Adam/Variable_46/AssignAssigntraining/Adam/Variable_46training/Adam/zeros_46*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_46
§
training/Adam/Variable_46/readIdentitytraining/Adam/Variable_46*
T0*,
_class"
 loc:@training/Adam/Variable_46*&
_output_shapes
: 
c
training/Adam/zeros_47Const*
valueB*    *
dtype0*
_output_shapes
:
Е
training/Adam/Variable_47
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ў
 training/Adam/Variable_47/AssignAssigntraining/Adam/Variable_47training/Adam/zeros_47*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_47*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_47/readIdentitytraining/Adam/Variable_47*
T0*,
_class"
 loc:@training/Adam/Variable_47*
_output_shapes
:
p
&training/Adam/zeros_48/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_48/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_48Fill&training/Adam/zeros_48/shape_as_tensortraining/Adam/zeros_48/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_48
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ў
 training/Adam/Variable_48/AssignAssigntraining/Adam/Variable_48training/Adam/zeros_48*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_48
Ш
training/Adam/Variable_48/readIdentitytraining/Adam/Variable_48*
T0*,
_class"
 loc:@training/Adam/Variable_48*
_output_shapes
:
p
&training/Adam/zeros_49/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_49/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_49Fill&training/Adam/zeros_49/shape_as_tensortraining/Adam/zeros_49/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_49
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
ў
 training/Adam/Variable_49/AssignAssigntraining/Adam/Variable_49training/Adam/zeros_49*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_49*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_49/readIdentitytraining/Adam/Variable_49*
T0*,
_class"
 loc:@training/Adam/Variable_49*
_output_shapes
:
p
&training/Adam/zeros_50/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_50/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_50Fill&training/Adam/zeros_50/shape_as_tensortraining/Adam/zeros_50/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_50
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
ў
 training/Adam/Variable_50/AssignAssigntraining/Adam/Variable_50training/Adam/zeros_50*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_50
Ш
training/Adam/Variable_50/readIdentitytraining/Adam/Variable_50*
T0*,
_class"
 loc:@training/Adam/Variable_50*
_output_shapes
:
p
&training/Adam/zeros_51/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_51/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ы
training/Adam/zeros_51Fill&training/Adam/zeros_51/shape_as_tensortraining/Adam/zeros_51/Const*
_output_shapes
:*
T0*

index_type0
Е
training/Adam/Variable_51
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
ў
 training/Adam/Variable_51/AssignAssigntraining/Adam/Variable_51training/Adam/zeros_51*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_51*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_51/readIdentitytraining/Adam/Variable_51*
T0*,
_class"
 loc:@training/Adam/Variable_51*
_output_shapes
:
p
&training/Adam/zeros_52/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_52/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_52Fill&training/Adam/zeros_52/shape_as_tensortraining/Adam/zeros_52/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_52
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
ў
 training/Adam/Variable_52/AssignAssigntraining/Adam/Variable_52training/Adam/zeros_52*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_52
Ш
training/Adam/Variable_52/readIdentitytraining/Adam/Variable_52*
T0*,
_class"
 loc:@training/Adam/Variable_52*
_output_shapes
:
p
&training/Adam/zeros_53/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_53/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ы
training/Adam/zeros_53Fill&training/Adam/zeros_53/shape_as_tensortraining/Adam/zeros_53/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_53
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ў
 training/Adam/Variable_53/AssignAssigntraining/Adam/Variable_53training/Adam/zeros_53*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_53
Ш
training/Adam/Variable_53/readIdentitytraining/Adam/Variable_53*
T0*,
_class"
 loc:@training/Adam/Variable_53*
_output_shapes
:
p
&training/Adam/zeros_54/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_54/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_54Fill&training/Adam/zeros_54/shape_as_tensortraining/Adam/zeros_54/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_54
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
ў
 training/Adam/Variable_54/AssignAssigntraining/Adam/Variable_54training/Adam/zeros_54*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_54
Ш
training/Adam/Variable_54/readIdentitytraining/Adam/Variable_54*
T0*,
_class"
 loc:@training/Adam/Variable_54*
_output_shapes
:
p
&training/Adam/zeros_55/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_55/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_55Fill&training/Adam/zeros_55/shape_as_tensortraining/Adam/zeros_55/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_55
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
ў
 training/Adam/Variable_55/AssignAssigntraining/Adam/Variable_55training/Adam/zeros_55*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_55
Ш
training/Adam/Variable_55/readIdentitytraining/Adam/Variable_55*
T0*,
_class"
 loc:@training/Adam/Variable_55*
_output_shapes
:
p
&training/Adam/zeros_56/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_56/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_56Fill&training/Adam/zeros_56/shape_as_tensortraining/Adam/zeros_56/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_56
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
ў
 training/Adam/Variable_56/AssignAssigntraining/Adam/Variable_56training/Adam/zeros_56*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_56*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_56/readIdentitytraining/Adam/Variable_56*
T0*,
_class"
 loc:@training/Adam/Variable_56*
_output_shapes
:
p
&training/Adam/zeros_57/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_57/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_57Fill&training/Adam/zeros_57/shape_as_tensortraining/Adam/zeros_57/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_57
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
ў
 training/Adam/Variable_57/AssignAssigntraining/Adam/Variable_57training/Adam/zeros_57*
T0*,
_class"
 loc:@training/Adam/Variable_57*
validate_shape(*
_output_shapes
:*
use_locking(
Ш
training/Adam/Variable_57/readIdentitytraining/Adam/Variable_57*
T0*,
_class"
 loc:@training/Adam/Variable_57*
_output_shapes
:
p
&training/Adam/zeros_58/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_58/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_58Fill&training/Adam/zeros_58/shape_as_tensortraining/Adam/zeros_58/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_58
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ў
 training/Adam/Variable_58/AssignAssigntraining/Adam/Variable_58training/Adam/zeros_58*
T0*,
_class"
 loc:@training/Adam/Variable_58*
validate_shape(*
_output_shapes
:*
use_locking(
Ш
training/Adam/Variable_58/readIdentitytraining/Adam/Variable_58*
T0*,
_class"
 loc:@training/Adam/Variable_58*
_output_shapes
:
p
&training/Adam/zeros_59/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_59/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_59Fill&training/Adam/zeros_59/shape_as_tensortraining/Adam/zeros_59/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_59
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
ў
 training/Adam/Variable_59/AssignAssigntraining/Adam/Variable_59training/Adam/zeros_59*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_59*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_59/readIdentitytraining/Adam/Variable_59*
T0*,
_class"
 loc:@training/Adam/Variable_59*
_output_shapes
:
p
&training/Adam/zeros_60/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_60/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_60Fill&training/Adam/zeros_60/shape_as_tensortraining/Adam/zeros_60/Const*
_output_shapes
:*
T0*

index_type0
Е
training/Adam/Variable_60
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ў
 training/Adam/Variable_60/AssignAssigntraining/Adam/Variable_60training/Adam/zeros_60*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_60*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_60/readIdentitytraining/Adam/Variable_60*
T0*,
_class"
 loc:@training/Adam/Variable_60*
_output_shapes
:
p
&training/Adam/zeros_61/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_61/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ы
training/Adam/zeros_61Fill&training/Adam/zeros_61/shape_as_tensortraining/Adam/zeros_61/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_61
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ў
 training/Adam/Variable_61/AssignAssigntraining/Adam/Variable_61training/Adam/zeros_61*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_61
Ш
training/Adam/Variable_61/readIdentitytraining/Adam/Variable_61*
T0*,
_class"
 loc:@training/Adam/Variable_61*
_output_shapes
:
p
&training/Adam/zeros_62/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_62/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_62Fill&training/Adam/zeros_62/shape_as_tensortraining/Adam/zeros_62/Const*
_output_shapes
:*
T0*

index_type0
Е
training/Adam/Variable_62
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ў
 training/Adam/Variable_62/AssignAssigntraining/Adam/Variable_62training/Adam/zeros_62*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_62*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_62/readIdentitytraining/Adam/Variable_62*
T0*,
_class"
 loc:@training/Adam/Variable_62*
_output_shapes
:
p
&training/Adam/zeros_63/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_63/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_63Fill&training/Adam/zeros_63/shape_as_tensortraining/Adam/zeros_63/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_63
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
ў
 training/Adam/Variable_63/AssignAssigntraining/Adam/Variable_63training/Adam/zeros_63*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_63
Ш
training/Adam/Variable_63/readIdentitytraining/Adam/Variable_63*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_63
p
&training/Adam/zeros_64/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_64/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_64Fill&training/Adam/zeros_64/shape_as_tensortraining/Adam/zeros_64/Const*
_output_shapes
:*
T0*

index_type0
Е
training/Adam/Variable_64
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ў
 training/Adam/Variable_64/AssignAssigntraining/Adam/Variable_64training/Adam/zeros_64*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_64*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_64/readIdentitytraining/Adam/Variable_64*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_64
p
&training/Adam/zeros_65/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_65/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_65Fill&training/Adam/zeros_65/shape_as_tensortraining/Adam/zeros_65/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_65
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
ў
 training/Adam/Variable_65/AssignAssigntraining/Adam/Variable_65training/Adam/zeros_65*
T0*,
_class"
 loc:@training/Adam/Variable_65*
validate_shape(*
_output_shapes
:*
use_locking(
Ш
training/Adam/Variable_65/readIdentitytraining/Adam/Variable_65*
_output_shapes
:*
T0*,
_class"
 loc:@training/Adam/Variable_65
p
&training/Adam/zeros_66/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_66/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_66Fill&training/Adam/zeros_66/shape_as_tensortraining/Adam/zeros_66/Const*
_output_shapes
:*
T0*

index_type0
Е
training/Adam/Variable_66
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
ў
 training/Adam/Variable_66/AssignAssigntraining/Adam/Variable_66training/Adam/zeros_66*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_66*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_66/readIdentitytraining/Adam/Variable_66*
T0*,
_class"
 loc:@training/Adam/Variable_66*
_output_shapes
:
p
&training/Adam/zeros_67/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_67/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_67Fill&training/Adam/zeros_67/shape_as_tensortraining/Adam/zeros_67/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_67
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ў
 training/Adam/Variable_67/AssignAssigntraining/Adam/Variable_67training/Adam/zeros_67*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_67*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_67/readIdentitytraining/Adam/Variable_67*
T0*,
_class"
 loc:@training/Adam/Variable_67*
_output_shapes
:
p
&training/Adam/zeros_68/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_68/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_68Fill&training/Adam/zeros_68/shape_as_tensortraining/Adam/zeros_68/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_68
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
ў
 training/Adam/Variable_68/AssignAssigntraining/Adam/Variable_68training/Adam/zeros_68*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_68*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_68/readIdentitytraining/Adam/Variable_68*
T0*,
_class"
 loc:@training/Adam/Variable_68*
_output_shapes
:
p
&training/Adam/zeros_69/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_69/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_69Fill&training/Adam/zeros_69/shape_as_tensortraining/Adam/zeros_69/Const*
_output_shapes
:*
T0*

index_type0
Е
training/Adam/Variable_69
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
ў
 training/Adam/Variable_69/AssignAssigntraining/Adam/Variable_69training/Adam/zeros_69*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_69*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_69/readIdentitytraining/Adam/Variable_69*
T0*,
_class"
 loc:@training/Adam/Variable_69*
_output_shapes
:
p
&training/Adam/zeros_70/shape_as_tensorConst*
valueB:*
dtype0*
_output_shapes
:
a
training/Adam/zeros_70/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_70Fill&training/Adam/zeros_70/shape_as_tensortraining/Adam/zeros_70/Const*
T0*

index_type0*
_output_shapes
:
Е
training/Adam/Variable_70
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
ў
 training/Adam/Variable_70/AssignAssigntraining/Adam/Variable_70training/Adam/zeros_70*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_70*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_70/readIdentitytraining/Adam/Variable_70*
T0*,
_class"
 loc:@training/Adam/Variable_70*
_output_shapes
:
p
&training/Adam/zeros_71/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:
a
training/Adam/zeros_71/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ы
training/Adam/zeros_71Fill&training/Adam/zeros_71/shape_as_tensortraining/Adam/zeros_71/Const*
_output_shapes
:*
T0*

index_type0
Е
training/Adam/Variable_71
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
ў
 training/Adam/Variable_71/AssignAssigntraining/Adam/Variable_71training/Adam/zeros_71*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_71*
validate_shape(*
_output_shapes
:
Ш
training/Adam/Variable_71/readIdentitytraining/Adam/Variable_71*
T0*,
_class"
 loc:@training/Adam/Variable_71*
_output_shapes
:
z
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*&
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 
±
training/Adam/mul_2Multraining/Adam/sub_2Otraining/Adam/gradients/down_level_0_no_0/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
u
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*&
_output_shapes
: 
}
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_24/read*
T0*&
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 
†
training/Adam/SquareSquareOtraining/Adam/gradients/down_level_0_no_0/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
v
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*&
_output_shapes
: 
u
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
T0*&
_output_shapes
: 
s
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
T0*&
_output_shapes
: 
Z
training/Adam/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_3Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_3*
T0*&
_output_shapes
: 
Ч
training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_2*
T0*&
_output_shapes
: 
l
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*&
_output_shapes
: 
Z
training/Adam/add_3/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
x
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*&
_output_shapes
: 
}
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*&
_output_shapes
: *
T0
Г
training/Adam/sub_4Subdown_level_0_no_0/kernel/readtraining/Adam/truediv_1*
T0*&
_output_shapes
: 
–
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
: *
use_locking(
Ў
training/Adam/Assign_1Assigntraining/Adam/Variable_24training/Adam/add_2*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24
÷
training/Adam/Assign_2Assigndown_level_0_no_0/kerneltraining/Adam/sub_4*
use_locking(*
T0*+
_class!
loc:@down_level_0_no_0/kernel*
validate_shape(*&
_output_shapes
: 
p
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes
: 
Z
training/Adam/sub_5/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0
Ш
training/Adam/mul_7Multraining/Adam/sub_5Btraining/Adam/gradients/down_level_0_no_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
i
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes
: 
q
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_25/read*
_output_shapes
: *
T0
Z
training/Adam/sub_6/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
T0*
_output_shapes
: 
Й
training/Adam/Square_1SquareBtraining/Adam/gradients/down_level_0_no_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
l
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
T0*
_output_shapes
: 
i
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes
: 
h
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes
: 
Z
training/Adam/Const_4Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_5Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Б
%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_5*
T0*
_output_shapes
: 
Л
training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_4*
_output_shapes
: *
T0
`
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes
: *
T0
Z
training/Adam/add_6/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
l
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes
: 
r
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes
: *
T0
u
training/Adam/sub_7Subdown_level_0_no_0/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes
: 
 
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
: 
ћ
training/Adam/Assign_4Assigntraining/Adam/Variable_25training/Adam/add_5*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
: 
∆
training/Adam/Assign_5Assigndown_level_0_no_0/biastraining/Adam/sub_7*
use_locking(*
T0*)
_class
loc:@down_level_0_no_0/bias*
validate_shape(*
_output_shapes
: 
}
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0*&
_output_shapes
:  
Z
training/Adam/sub_8/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
T0*
_output_shapes
: 
≤
training/Adam/mul_12Multraining/Adam/sub_8Otraining/Adam/gradients/down_level_0_no_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:  *
T0
w
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0*&
_output_shapes
:  
~
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_26/read*&
_output_shapes
:  *
T0
Z
training/Adam/sub_9/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
_output_shapes
: *
T0
Ґ
training/Adam/Square_2SquareOtraining/Adam/gradients/down_level_0_no_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:  
y
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0*&
_output_shapes
:  
w
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0*&
_output_shapes
:  
t
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*&
_output_shapes
:  *
T0
Z
training/Adam/Const_6Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_7Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Н
%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_7*&
_output_shapes
:  *
T0
Ч
training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_6*
T0*&
_output_shapes
:  
l
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0*&
_output_shapes
:  
Z
training/Adam/add_9/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
x
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0*&
_output_shapes
:  
~
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*&
_output_shapes
:  *
T0
Д
training/Adam/sub_10Subdown_level_0_no_1/kernel/readtraining/Adam/truediv_3*
T0*&
_output_shapes
:  
÷
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
:  
Ў
training/Adam/Assign_7Assigntraining/Adam/Variable_26training/Adam/add_8*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_26
„
training/Adam/Assign_8Assigndown_level_0_no_1/kerneltraining/Adam/sub_10*
use_locking(*
T0*+
_class!
loc:@down_level_0_no_1/kernel*
validate_shape(*&
_output_shapes
:  
q
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes
: 
[
training/Adam/sub_11/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ъ
training/Adam/mul_17Multraining/Adam/sub_11Btraining/Adam/gradients/down_level_0_no_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
l
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
_output_shapes
: *
T0
r
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_27/read*
T0*
_output_shapes
: 
[
training/Adam/sub_12/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
_output_shapes
: *
T0
Й
training/Adam/Square_3SquareBtraining/Adam/gradients/down_level_0_no_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
n
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
_output_shapes
: *
T0
l
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes
: 
i
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
_output_shapes
: *
T0
Z
training/Adam/Const_8Const*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
training/Adam/Const_9Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
В
%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_9*
T0*
_output_shapes
: 
Л
training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_8*
T0*
_output_shapes
: 
`
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
_output_shapes
: *
T0
[
training/Adam/add_12/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
n
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
_output_shapes
: *
T0
s
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes
: 
v
training/Adam/sub_13Subdown_level_0_no_1/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes
: 
Ћ
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
: 
ќ
training/Adam/Assign_10Assigntraining/Adam/Variable_27training/Adam/add_11*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27
»
training/Adam/Assign_11Assigndown_level_0_no_1/biastraining/Adam/sub_13*
T0*)
_class
loc:@down_level_0_no_1/bias*
validate_shape(*
_output_shapes
: *
use_locking(
}
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0*&
_output_shapes
: @
[
training/Adam/sub_14/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 
≥
training/Adam/mul_22Multraining/Adam/sub_14Otraining/Adam/gradients/down_level_1_no_0/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
x
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0*&
_output_shapes
: @
~
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_28/read*
T0*&
_output_shapes
: @
[
training/Adam/sub_15/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
_output_shapes
: *
T0
Ґ
training/Adam/Square_4SquareOtraining/Adam/gradients/down_level_1_no_0/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: @
z
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0*&
_output_shapes
: @
x
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0*&
_output_shapes
: @
u
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0*&
_output_shapes
: @
[
training/Adam/Const_10Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_11Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
П
%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_11*
T0*&
_output_shapes
: @
Ш
training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_10*
T0*&
_output_shapes
: @
l
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0*&
_output_shapes
: @
[
training/Adam/add_15/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
z
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0*&
_output_shapes
: @

training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15*
T0*&
_output_shapes
: @
Д
training/Adam/sub_16Subdown_level_1_no_0/kernel/readtraining/Adam/truediv_5*&
_output_shapes
: @*
T0
Ў
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4
Џ
training/Adam/Assign_13Assigntraining/Adam/Variable_28training/Adam/add_14*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_28*
validate_shape(*&
_output_shapes
: @
Ў
training/Adam/Assign_14Assigndown_level_1_no_0/kerneltraining/Adam/sub_16*
use_locking(*
T0*+
_class!
loc:@down_level_1_no_0/kernel*
validate_shape(*&
_output_shapes
: @
q
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
_output_shapes
:@*
T0
[
training/Adam/sub_17/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ъ
training/Adam/mul_27Multraining/Adam/sub_17Btraining/Adam/gradients/down_level_1_no_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
l
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
T0*
_output_shapes
:@
r
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_29/read*
T0*
_output_shapes
:@
[
training/Adam/sub_18/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 
Й
training/Adam/Square_5SquareBtraining/Adam/gradients/down_level_1_no_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
n
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes
:@
l
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
_output_shapes
:@*
T0
i
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
T0*
_output_shapes
:@
[
training/Adam/Const_12Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_13Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Г
%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_13*
T0*
_output_shapes
:@
М
training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_12*
T0*
_output_shapes
:@
`
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes
:@
[
training/Adam/add_18/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
n
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes
:@
s
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
_output_shapes
:@*
T0
v
training/Adam/sub_19Subdown_level_1_no_0/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes
:@
ћ
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:@
ќ
training/Adam/Assign_16Assigntraining/Adam/Variable_29training/Adam/add_17*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_29
»
training/Adam/Assign_17Assigndown_level_1_no_0/biastraining/Adam/sub_19*
use_locking(*
T0*)
_class
loc:@down_level_1_no_0/bias*
validate_shape(*
_output_shapes
:@
}
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*&
_output_shapes
:@@*
T0
[
training/Adam/sub_20/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
_output_shapes
: *
T0
≥
training/Adam/mul_32Multraining/Adam/sub_20Otraining/Adam/gradients/down_level_1_no_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
x
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*&
_output_shapes
:@@
~
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_30/read*&
_output_shapes
:@@*
T0
[
training/Adam/sub_21/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 
Ґ
training/Adam/Square_6SquareOtraining/Adam/gradients/down_level_1_no_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
z
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*&
_output_shapes
:@@
x
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*&
_output_shapes
:@@
u
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*&
_output_shapes
:@@
[
training/Adam/Const_14Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_15Const*
dtype0*
_output_shapes
: *
valueB
 *  А
П
%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_15*
T0*&
_output_shapes
:@@
Ш
training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_14*&
_output_shapes
:@@*
T0
l
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*&
_output_shapes
:@@*
T0
[
training/Adam/add_21/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
z
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*&
_output_shapes
:@@

training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*&
_output_shapes
:@@
Д
training/Adam/sub_22Subdown_level_1_no_1/kernel/readtraining/Adam/truediv_7*
T0*&
_output_shapes
:@@
Ў
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_6
Џ
training/Adam/Assign_19Assigntraining/Adam/Variable_30training/Adam/add_20*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_30*
validate_shape(*&
_output_shapes
:@@
Ў
training/Adam/Assign_20Assigndown_level_1_no_1/kerneltraining/Adam/sub_22*
T0*+
_class!
loc:@down_level_1_no_1/kernel*
validate_shape(*&
_output_shapes
:@@*
use_locking(
q
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
T0*
_output_shapes
:@
[
training/Adam/sub_23/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ъ
training/Adam/mul_37Multraining/Adam/sub_23Btraining/Adam/gradients/down_level_1_no_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
l
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes
:@
r
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_31/read*
T0*
_output_shapes
:@
[
training/Adam/sub_24/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
_output_shapes
: *
T0
Й
training/Adam/Square_7SquareBtraining/Adam/gradients/down_level_1_no_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes
:@
l
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes
:@
i
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
_output_shapes
:@*
T0
[
training/Adam/Const_16Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_17Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Г
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_17*
T0*
_output_shapes
:@
М
training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_16*
_output_shapes
:@*
T0
`
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
_output_shapes
:@*
T0
[
training/Adam/add_24/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
n
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
T0*
_output_shapes
:@
s
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
_output_shapes
:@*
T0
v
training/Adam/sub_25Subdown_level_1_no_1/bias/readtraining/Adam/truediv_8*
_output_shapes
:@*
T0
ћ
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
:@*
use_locking(
ќ
training/Adam/Assign_22Assigntraining/Adam/Variable_31training/Adam/add_23*
T0*,
_class"
 loc:@training/Adam/Variable_31*
validate_shape(*
_output_shapes
:@*
use_locking(
»
training/Adam/Assign_23Assigndown_level_1_no_1/biastraining/Adam/sub_25*
use_locking(*
T0*)
_class
loc:@down_level_1_no_1/bias*
validate_shape(*
_output_shapes
:@
~
training/Adam/mul_41MulAdam/beta_1/readtraining/Adam/Variable_8/read*
T0*'
_output_shapes
:@А
[
training/Adam/sub_26/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_26Subtraining/Adam/sub_26/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ђ
training/Adam/mul_42Multraining/Adam/sub_26Ftraining/Adam/gradients/middle_0/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@А
y
training/Adam/add_25Addtraining/Adam/mul_41training/Adam/mul_42*'
_output_shapes
:@А*
T0

training/Adam/mul_43MulAdam/beta_2/readtraining/Adam/Variable_32/read*
T0*'
_output_shapes
:@А
[
training/Adam/sub_27/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
f
training/Adam/sub_27Subtraining/Adam/sub_27/xAdam/beta_2/read*
T0*
_output_shapes
: 
Ъ
training/Adam/Square_8SquareFtraining/Adam/gradients/middle_0/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:@А
{
training/Adam/mul_44Multraining/Adam/sub_27training/Adam/Square_8*
T0*'
_output_shapes
:@А
y
training/Adam/add_26Addtraining/Adam/mul_43training/Adam/mul_44*
T0*'
_output_shapes
:@А
v
training/Adam/mul_45Multraining/Adam/multraining/Adam/add_25*
T0*'
_output_shapes
:@А
[
training/Adam/Const_18Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_19Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Р
%training/Adam/clip_by_value_9/MinimumMinimumtraining/Adam/add_26training/Adam/Const_19*
T0*'
_output_shapes
:@А
Щ
training/Adam/clip_by_value_9Maximum%training/Adam/clip_by_value_9/Minimumtraining/Adam/Const_18*
T0*'
_output_shapes
:@А
m
training/Adam/Sqrt_9Sqrttraining/Adam/clip_by_value_9*
T0*'
_output_shapes
:@А
[
training/Adam/add_27/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
{
training/Adam/add_27Addtraining/Adam/Sqrt_9training/Adam/add_27/y*
T0*'
_output_shapes
:@А
А
training/Adam/truediv_9RealDivtraining/Adam/mul_45training/Adam/add_27*
T0*'
_output_shapes
:@А
|
training/Adam/sub_28Submiddle_0/kernel/readtraining/Adam/truediv_9*
T0*'
_output_shapes
:@А
ў
training/Adam/Assign_24Assigntraining/Adam/Variable_8training/Adam/add_25*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*'
_output_shapes
:@А
џ
training/Adam/Assign_25Assigntraining/Adam/Variable_32training/Adam/add_26*
T0*,
_class"
 loc:@training/Adam/Variable_32*
validate_shape(*'
_output_shapes
:@А*
use_locking(
«
training/Adam/Assign_26Assignmiddle_0/kerneltraining/Adam/sub_28*
use_locking(*
T0*"
_class
loc:@middle_0/kernel*
validate_shape(*'
_output_shapes
:@А
r
training/Adam/mul_46MulAdam/beta_1/readtraining/Adam/Variable_9/read*
T0*
_output_shapes	
:А
[
training/Adam/sub_29/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_29Subtraining/Adam/sub_29/xAdam/beta_1/read*
T0*
_output_shapes
: 
Т
training/Adam/mul_47Multraining/Adam/sub_299training/Adam/gradients/middle_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:А
m
training/Adam/add_28Addtraining/Adam/mul_46training/Adam/mul_47*
T0*
_output_shapes	
:А
s
training/Adam/mul_48MulAdam/beta_2/readtraining/Adam/Variable_33/read*
T0*
_output_shapes	
:А
[
training/Adam/sub_30/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_30Subtraining/Adam/sub_30/xAdam/beta_2/read*
T0*
_output_shapes
: 
Б
training/Adam/Square_9Square9training/Adam/gradients/middle_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:А
o
training/Adam/mul_49Multraining/Adam/sub_30training/Adam/Square_9*
T0*
_output_shapes	
:А
m
training/Adam/add_29Addtraining/Adam/mul_48training/Adam/mul_49*
T0*
_output_shapes	
:А
j
training/Adam/mul_50Multraining/Adam/multraining/Adam/add_28*
T0*
_output_shapes	
:А
[
training/Adam/Const_20Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_21Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Е
&training/Adam/clip_by_value_10/MinimumMinimumtraining/Adam/add_29training/Adam/Const_21*
T0*
_output_shapes	
:А
П
training/Adam/clip_by_value_10Maximum&training/Adam/clip_by_value_10/Minimumtraining/Adam/Const_20*
_output_shapes	
:А*
T0
c
training/Adam/Sqrt_10Sqrttraining/Adam/clip_by_value_10*
_output_shapes	
:А*
T0
[
training/Adam/add_30/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
p
training/Adam/add_30Addtraining/Adam/Sqrt_10training/Adam/add_30/y*
T0*
_output_shapes	
:А
u
training/Adam/truediv_10RealDivtraining/Adam/mul_50training/Adam/add_30*
_output_shapes	
:А*
T0
o
training/Adam/sub_31Submiddle_0/bias/readtraining/Adam/truediv_10*
T0*
_output_shapes	
:А
Ќ
training/Adam/Assign_27Assigntraining/Adam/Variable_9training/Adam/add_28*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
_output_shapes	
:А
ѕ
training/Adam/Assign_28Assigntraining/Adam/Variable_33training/Adam/add_29*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_33
Ј
training/Adam/Assign_29Assignmiddle_0/biastraining/Adam/sub_31*
T0* 
_class
loc:@middle_0/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(

training/Adam/mul_51MulAdam/beta_1/readtraining/Adam/Variable_10/read*
T0*'
_output_shapes
:А@
[
training/Adam/sub_32/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_32Subtraining/Adam/sub_32/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ђ
training/Adam/mul_52Multraining/Adam/sub_32Ftraining/Adam/gradients/middle_2/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:А@
y
training/Adam/add_31Addtraining/Adam/mul_51training/Adam/mul_52*
T0*'
_output_shapes
:А@

training/Adam/mul_53MulAdam/beta_2/readtraining/Adam/Variable_34/read*
T0*'
_output_shapes
:А@
[
training/Adam/sub_33/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
f
training/Adam/sub_33Subtraining/Adam/sub_33/xAdam/beta_2/read*
T0*
_output_shapes
: 
Ы
training/Adam/Square_10SquareFtraining/Adam/gradients/middle_2/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:А@
|
training/Adam/mul_54Multraining/Adam/sub_33training/Adam/Square_10*'
_output_shapes
:А@*
T0
y
training/Adam/add_32Addtraining/Adam/mul_53training/Adam/mul_54*
T0*'
_output_shapes
:А@
v
training/Adam/mul_55Multraining/Adam/multraining/Adam/add_31*
T0*'
_output_shapes
:А@
[
training/Adam/Const_22Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_23Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
С
&training/Adam/clip_by_value_11/MinimumMinimumtraining/Adam/add_32training/Adam/Const_23*
T0*'
_output_shapes
:А@
Ы
training/Adam/clip_by_value_11Maximum&training/Adam/clip_by_value_11/Minimumtraining/Adam/Const_22*
T0*'
_output_shapes
:А@
o
training/Adam/Sqrt_11Sqrttraining/Adam/clip_by_value_11*'
_output_shapes
:А@*
T0
[
training/Adam/add_33/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
|
training/Adam/add_33Addtraining/Adam/Sqrt_11training/Adam/add_33/y*
T0*'
_output_shapes
:А@
Б
training/Adam/truediv_11RealDivtraining/Adam/mul_55training/Adam/add_33*'
_output_shapes
:А@*
T0
}
training/Adam/sub_34Submiddle_2/kernel/readtraining/Adam/truediv_11*
T0*'
_output_shapes
:А@
џ
training/Adam/Assign_30Assigntraining/Adam/Variable_10training/Adam/add_31*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*'
_output_shapes
:А@
џ
training/Adam/Assign_31Assigntraining/Adam/Variable_34training/Adam/add_32*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_34*
validate_shape(*'
_output_shapes
:А@
«
training/Adam/Assign_32Assignmiddle_2/kerneltraining/Adam/sub_34*
use_locking(*
T0*"
_class
loc:@middle_2/kernel*
validate_shape(*'
_output_shapes
:А@
r
training/Adam/mul_56MulAdam/beta_1/readtraining/Adam/Variable_11/read*
T0*
_output_shapes
:@
[
training/Adam/sub_35/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_35Subtraining/Adam/sub_35/xAdam/beta_1/read*
_output_shapes
: *
T0
С
training/Adam/mul_57Multraining/Adam/sub_359training/Adam/gradients/middle_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
l
training/Adam/add_34Addtraining/Adam/mul_56training/Adam/mul_57*
T0*
_output_shapes
:@
r
training/Adam/mul_58MulAdam/beta_2/readtraining/Adam/Variable_35/read*
_output_shapes
:@*
T0
[
training/Adam/sub_36/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
f
training/Adam/sub_36Subtraining/Adam/sub_36/xAdam/beta_2/read*
T0*
_output_shapes
: 
Б
training/Adam/Square_11Square9training/Adam/gradients/middle_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
o
training/Adam/mul_59Multraining/Adam/sub_36training/Adam/Square_11*
T0*
_output_shapes
:@
l
training/Adam/add_35Addtraining/Adam/mul_58training/Adam/mul_59*
_output_shapes
:@*
T0
i
training/Adam/mul_60Multraining/Adam/multraining/Adam/add_34*
T0*
_output_shapes
:@
[
training/Adam/Const_24Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_25Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Д
&training/Adam/clip_by_value_12/MinimumMinimumtraining/Adam/add_35training/Adam/Const_25*
T0*
_output_shapes
:@
О
training/Adam/clip_by_value_12Maximum&training/Adam/clip_by_value_12/Minimumtraining/Adam/Const_24*
T0*
_output_shapes
:@
b
training/Adam/Sqrt_12Sqrttraining/Adam/clip_by_value_12*
T0*
_output_shapes
:@
[
training/Adam/add_36/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
o
training/Adam/add_36Addtraining/Adam/Sqrt_12training/Adam/add_36/y*
T0*
_output_shapes
:@
t
training/Adam/truediv_12RealDivtraining/Adam/mul_60training/Adam/add_36*
T0*
_output_shapes
:@
n
training/Adam/sub_37Submiddle_2/bias/readtraining/Adam/truediv_12*
T0*
_output_shapes
:@
ќ
training/Adam/Assign_33Assigntraining/Adam/Variable_11training/Adam/add_34*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11
ќ
training/Adam/Assign_34Assigntraining/Adam/Variable_35training/Adam/add_35*
T0*,
_class"
 loc:@training/Adam/Variable_35*
validate_shape(*
_output_shapes
:@*
use_locking(
ґ
training/Adam/Assign_35Assignmiddle_2/biastraining/Adam/sub_37*
use_locking(*
T0* 
_class
loc:@middle_2/bias*
validate_shape(*
_output_shapes
:@

training/Adam/mul_61MulAdam/beta_1/readtraining/Adam/Variable_12/read*
T0*'
_output_shapes
:А@
[
training/Adam/sub_38/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_38Subtraining/Adam/sub_38/xAdam/beta_1/read*
T0*
_output_shapes
: 
≤
training/Adam/mul_62Multraining/Adam/sub_38Mtraining/Adam/gradients/up_level_1_no_0/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:А@
y
training/Adam/add_37Addtraining/Adam/mul_61training/Adam/mul_62*
T0*'
_output_shapes
:А@

training/Adam/mul_63MulAdam/beta_2/readtraining/Adam/Variable_36/read*
T0*'
_output_shapes
:А@
[
training/Adam/sub_39/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_39Subtraining/Adam/sub_39/xAdam/beta_2/read*
T0*
_output_shapes
: 
Ґ
training/Adam/Square_12SquareMtraining/Adam/gradients/up_level_1_no_0/convolution_grad/Conv2DBackpropFilter*
T0*'
_output_shapes
:А@
|
training/Adam/mul_64Multraining/Adam/sub_39training/Adam/Square_12*
T0*'
_output_shapes
:А@
y
training/Adam/add_38Addtraining/Adam/mul_63training/Adam/mul_64*
T0*'
_output_shapes
:А@
v
training/Adam/mul_65Multraining/Adam/multraining/Adam/add_37*
T0*'
_output_shapes
:А@
[
training/Adam/Const_26Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_27Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
С
&training/Adam/clip_by_value_13/MinimumMinimumtraining/Adam/add_38training/Adam/Const_27*'
_output_shapes
:А@*
T0
Ы
training/Adam/clip_by_value_13Maximum&training/Adam/clip_by_value_13/Minimumtraining/Adam/Const_26*
T0*'
_output_shapes
:А@
o
training/Adam/Sqrt_13Sqrttraining/Adam/clip_by_value_13*'
_output_shapes
:А@*
T0
[
training/Adam/add_39/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
|
training/Adam/add_39Addtraining/Adam/Sqrt_13training/Adam/add_39/y*
T0*'
_output_shapes
:А@
Б
training/Adam/truediv_13RealDivtraining/Adam/mul_65training/Adam/add_39*'
_output_shapes
:А@*
T0
Д
training/Adam/sub_40Subup_level_1_no_0/kernel/readtraining/Adam/truediv_13*'
_output_shapes
:А@*
T0
џ
training/Adam/Assign_36Assigntraining/Adam/Variable_12training/Adam/add_37*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*'
_output_shapes
:А@*
use_locking(
џ
training/Adam/Assign_37Assigntraining/Adam/Variable_36training/Adam/add_38*
validate_shape(*'
_output_shapes
:А@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_36
’
training/Adam/Assign_38Assignup_level_1_no_0/kerneltraining/Adam/sub_40*
use_locking(*
T0*)
_class
loc:@up_level_1_no_0/kernel*
validate_shape(*'
_output_shapes
:А@
r
training/Adam/mul_66MulAdam/beta_1/readtraining/Adam/Variable_13/read*
T0*
_output_shapes
:@
[
training/Adam/sub_41/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_41Subtraining/Adam/sub_41/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ш
training/Adam/mul_67Multraining/Adam/sub_41@training/Adam/gradients/up_level_1_no_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
l
training/Adam/add_40Addtraining/Adam/mul_66training/Adam/mul_67*
T0*
_output_shapes
:@
r
training/Adam/mul_68MulAdam/beta_2/readtraining/Adam/Variable_37/read*
T0*
_output_shapes
:@
[
training/Adam/sub_42/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_42Subtraining/Adam/sub_42/xAdam/beta_2/read*
_output_shapes
: *
T0
И
training/Adam/Square_13Square@training/Adam/gradients/up_level_1_no_0/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
o
training/Adam/mul_69Multraining/Adam/sub_42training/Adam/Square_13*
T0*
_output_shapes
:@
l
training/Adam/add_41Addtraining/Adam/mul_68training/Adam/mul_69*
T0*
_output_shapes
:@
i
training/Adam/mul_70Multraining/Adam/multraining/Adam/add_40*
T0*
_output_shapes
:@
[
training/Adam/Const_28Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_29Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Д
&training/Adam/clip_by_value_14/MinimumMinimumtraining/Adam/add_41training/Adam/Const_29*
T0*
_output_shapes
:@
О
training/Adam/clip_by_value_14Maximum&training/Adam/clip_by_value_14/Minimumtraining/Adam/Const_28*
T0*
_output_shapes
:@
b
training/Adam/Sqrt_14Sqrttraining/Adam/clip_by_value_14*
T0*
_output_shapes
:@
[
training/Adam/add_42/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
o
training/Adam/add_42Addtraining/Adam/Sqrt_14training/Adam/add_42/y*
_output_shapes
:@*
T0
t
training/Adam/truediv_14RealDivtraining/Adam/mul_70training/Adam/add_42*
_output_shapes
:@*
T0
u
training/Adam/sub_43Subup_level_1_no_0/bias/readtraining/Adam/truediv_14*
T0*
_output_shapes
:@
ќ
training/Adam/Assign_39Assigntraining/Adam/Variable_13training/Adam/add_40*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:@*
use_locking(
ќ
training/Adam/Assign_40Assigntraining/Adam/Variable_37training/Adam/add_41*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_37
ƒ
training/Adam/Assign_41Assignup_level_1_no_0/biastraining/Adam/sub_43*
use_locking(*
T0*'
_class
loc:@up_level_1_no_0/bias*
validate_shape(*
_output_shapes
:@
~
training/Adam/mul_71MulAdam/beta_1/readtraining/Adam/Variable_14/read*
T0*&
_output_shapes
:@ 
[
training/Adam/sub_44/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
f
training/Adam/sub_44Subtraining/Adam/sub_44/xAdam/beta_1/read*
T0*
_output_shapes
: 
±
training/Adam/mul_72Multraining/Adam/sub_44Mtraining/Adam/gradients/up_level_1_no_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@ *
T0
x
training/Adam/add_43Addtraining/Adam/mul_71training/Adam/mul_72*
T0*&
_output_shapes
:@ 
~
training/Adam/mul_73MulAdam/beta_2/readtraining/Adam/Variable_38/read*
T0*&
_output_shapes
:@ 
[
training/Adam/sub_45/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
f
training/Adam/sub_45Subtraining/Adam/sub_45/xAdam/beta_2/read*
T0*
_output_shapes
: 
°
training/Adam/Square_14SquareMtraining/Adam/gradients/up_level_1_no_2/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@ 
{
training/Adam/mul_74Multraining/Adam/sub_45training/Adam/Square_14*
T0*&
_output_shapes
:@ 
x
training/Adam/add_44Addtraining/Adam/mul_73training/Adam/mul_74*
T0*&
_output_shapes
:@ 
u
training/Adam/mul_75Multraining/Adam/multraining/Adam/add_43*
T0*&
_output_shapes
:@ 
[
training/Adam/Const_30Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_31Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Р
&training/Adam/clip_by_value_15/MinimumMinimumtraining/Adam/add_44training/Adam/Const_31*&
_output_shapes
:@ *
T0
Ъ
training/Adam/clip_by_value_15Maximum&training/Adam/clip_by_value_15/Minimumtraining/Adam/Const_30*
T0*&
_output_shapes
:@ 
n
training/Adam/Sqrt_15Sqrttraining/Adam/clip_by_value_15*
T0*&
_output_shapes
:@ 
[
training/Adam/add_45/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
{
training/Adam/add_45Addtraining/Adam/Sqrt_15training/Adam/add_45/y*
T0*&
_output_shapes
:@ 
А
training/Adam/truediv_15RealDivtraining/Adam/mul_75training/Adam/add_45*
T0*&
_output_shapes
:@ 
Г
training/Adam/sub_46Subup_level_1_no_2/kernel/readtraining/Adam/truediv_15*
T0*&
_output_shapes
:@ 
Џ
training/Adam/Assign_42Assigntraining/Adam/Variable_14training/Adam/add_43*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*&
_output_shapes
:@ *
use_locking(
Џ
training/Adam/Assign_43Assigntraining/Adam/Variable_38training/Adam/add_44*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_38
‘
training/Adam/Assign_44Assignup_level_1_no_2/kerneltraining/Adam/sub_46*
use_locking(*
T0*)
_class
loc:@up_level_1_no_2/kernel*
validate_shape(*&
_output_shapes
:@ 
r
training/Adam/mul_76MulAdam/beta_1/readtraining/Adam/Variable_15/read*
T0*
_output_shapes
: 
[
training/Adam/sub_47/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_47Subtraining/Adam/sub_47/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ш
training/Adam/mul_77Multraining/Adam/sub_47@training/Adam/gradients/up_level_1_no_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
l
training/Adam/add_46Addtraining/Adam/mul_76training/Adam/mul_77*
T0*
_output_shapes
: 
r
training/Adam/mul_78MulAdam/beta_2/readtraining/Adam/Variable_39/read*
T0*
_output_shapes
: 
[
training/Adam/sub_48/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_48Subtraining/Adam/sub_48/xAdam/beta_2/read*
T0*
_output_shapes
: 
И
training/Adam/Square_15Square@training/Adam/gradients/up_level_1_no_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
o
training/Adam/mul_79Multraining/Adam/sub_48training/Adam/Square_15*
T0*
_output_shapes
: 
l
training/Adam/add_47Addtraining/Adam/mul_78training/Adam/mul_79*
_output_shapes
: *
T0
i
training/Adam/mul_80Multraining/Adam/multraining/Adam/add_46*
T0*
_output_shapes
: 
[
training/Adam/Const_32Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_33Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Д
&training/Adam/clip_by_value_16/MinimumMinimumtraining/Adam/add_47training/Adam/Const_33*
T0*
_output_shapes
: 
О
training/Adam/clip_by_value_16Maximum&training/Adam/clip_by_value_16/Minimumtraining/Adam/Const_32*
T0*
_output_shapes
: 
b
training/Adam/Sqrt_16Sqrttraining/Adam/clip_by_value_16*
_output_shapes
: *
T0
[
training/Adam/add_48/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
o
training/Adam/add_48Addtraining/Adam/Sqrt_16training/Adam/add_48/y*
_output_shapes
: *
T0
t
training/Adam/truediv_16RealDivtraining/Adam/mul_80training/Adam/add_48*
_output_shapes
: *
T0
u
training/Adam/sub_49Subup_level_1_no_2/bias/readtraining/Adam/truediv_16*
T0*
_output_shapes
: 
ќ
training/Adam/Assign_45Assigntraining/Adam/Variable_15training/Adam/add_46*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
: 
ќ
training/Adam/Assign_46Assigntraining/Adam/Variable_39training/Adam/add_47*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_39*
validate_shape(*
_output_shapes
: 
ƒ
training/Adam/Assign_47Assignup_level_1_no_2/biastraining/Adam/sub_49*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@up_level_1_no_2/bias
~
training/Adam/mul_81MulAdam/beta_1/readtraining/Adam/Variable_16/read*&
_output_shapes
:@ *
T0
[
training/Adam/sub_50/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_50Subtraining/Adam/sub_50/xAdam/beta_1/read*
T0*
_output_shapes
: 
±
training/Adam/mul_82Multraining/Adam/sub_50Mtraining/Adam/gradients/up_level_0_no_0/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@ *
T0
x
training/Adam/add_49Addtraining/Adam/mul_81training/Adam/mul_82*
T0*&
_output_shapes
:@ 
~
training/Adam/mul_83MulAdam/beta_2/readtraining/Adam/Variable_40/read*
T0*&
_output_shapes
:@ 
[
training/Adam/sub_51/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_51Subtraining/Adam/sub_51/xAdam/beta_2/read*
T0*
_output_shapes
: 
°
training/Adam/Square_16SquareMtraining/Adam/gradients/up_level_0_no_0/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@ 
{
training/Adam/mul_84Multraining/Adam/sub_51training/Adam/Square_16*
T0*&
_output_shapes
:@ 
x
training/Adam/add_50Addtraining/Adam/mul_83training/Adam/mul_84*
T0*&
_output_shapes
:@ 
u
training/Adam/mul_85Multraining/Adam/multraining/Adam/add_49*
T0*&
_output_shapes
:@ 
[
training/Adam/Const_34Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_35Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Р
&training/Adam/clip_by_value_17/MinimumMinimumtraining/Adam/add_50training/Adam/Const_35*
T0*&
_output_shapes
:@ 
Ъ
training/Adam/clip_by_value_17Maximum&training/Adam/clip_by_value_17/Minimumtraining/Adam/Const_34*
T0*&
_output_shapes
:@ 
n
training/Adam/Sqrt_17Sqrttraining/Adam/clip_by_value_17*
T0*&
_output_shapes
:@ 
[
training/Adam/add_51/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
{
training/Adam/add_51Addtraining/Adam/Sqrt_17training/Adam/add_51/y*
T0*&
_output_shapes
:@ 
А
training/Adam/truediv_17RealDivtraining/Adam/mul_85training/Adam/add_51*
T0*&
_output_shapes
:@ 
Г
training/Adam/sub_52Subup_level_0_no_0/kernel/readtraining/Adam/truediv_17*
T0*&
_output_shapes
:@ 
Џ
training/Adam/Assign_48Assigntraining/Adam/Variable_16training/Adam/add_49*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16
Џ
training/Adam/Assign_49Assigntraining/Adam/Variable_40training/Adam/add_50*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_40
‘
training/Adam/Assign_50Assignup_level_0_no_0/kerneltraining/Adam/sub_52*
T0*)
_class
loc:@up_level_0_no_0/kernel*
validate_shape(*&
_output_shapes
:@ *
use_locking(
r
training/Adam/mul_86MulAdam/beta_1/readtraining/Adam/Variable_17/read*
_output_shapes
: *
T0
[
training/Adam/sub_53/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
f
training/Adam/sub_53Subtraining/Adam/sub_53/xAdam/beta_1/read*
_output_shapes
: *
T0
Ш
training/Adam/mul_87Multraining/Adam/sub_53@training/Adam/gradients/up_level_0_no_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
l
training/Adam/add_52Addtraining/Adam/mul_86training/Adam/mul_87*
T0*
_output_shapes
: 
r
training/Adam/mul_88MulAdam/beta_2/readtraining/Adam/Variable_41/read*
T0*
_output_shapes
: 
[
training/Adam/sub_54/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_54Subtraining/Adam/sub_54/xAdam/beta_2/read*
T0*
_output_shapes
: 
И
training/Adam/Square_17Square@training/Adam/gradients/up_level_0_no_0/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
o
training/Adam/mul_89Multraining/Adam/sub_54training/Adam/Square_17*
_output_shapes
: *
T0
l
training/Adam/add_53Addtraining/Adam/mul_88training/Adam/mul_89*
T0*
_output_shapes
: 
i
training/Adam/mul_90Multraining/Adam/multraining/Adam/add_52*
T0*
_output_shapes
: 
[
training/Adam/Const_36Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_37Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Д
&training/Adam/clip_by_value_18/MinimumMinimumtraining/Adam/add_53training/Adam/Const_37*
T0*
_output_shapes
: 
О
training/Adam/clip_by_value_18Maximum&training/Adam/clip_by_value_18/Minimumtraining/Adam/Const_36*
_output_shapes
: *
T0
b
training/Adam/Sqrt_18Sqrttraining/Adam/clip_by_value_18*
_output_shapes
: *
T0
[
training/Adam/add_54/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
o
training/Adam/add_54Addtraining/Adam/Sqrt_18training/Adam/add_54/y*
T0*
_output_shapes
: 
t
training/Adam/truediv_18RealDivtraining/Adam/mul_90training/Adam/add_54*
T0*
_output_shapes
: 
u
training/Adam/sub_55Subup_level_0_no_0/bias/readtraining/Adam/truediv_18*
T0*
_output_shapes
: 
ќ
training/Adam/Assign_51Assigntraining/Adam/Variable_17training/Adam/add_52*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
: 
ќ
training/Adam/Assign_52Assigntraining/Adam/Variable_41training/Adam/add_53*
T0*,
_class"
 loc:@training/Adam/Variable_41*
validate_shape(*
_output_shapes
: *
use_locking(
ƒ
training/Adam/Assign_53Assignup_level_0_no_0/biastraining/Adam/sub_55*
use_locking(*
T0*'
_class
loc:@up_level_0_no_0/bias*
validate_shape(*
_output_shapes
: 
~
training/Adam/mul_91MulAdam/beta_1/readtraining/Adam/Variable_18/read*
T0*&
_output_shapes
:  
[
training/Adam/sub_56/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_56Subtraining/Adam/sub_56/xAdam/beta_1/read*
_output_shapes
: *
T0
±
training/Adam/mul_92Multraining/Adam/sub_56Mtraining/Adam/gradients/up_level_0_no_2/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:  
x
training/Adam/add_55Addtraining/Adam/mul_91training/Adam/mul_92*
T0*&
_output_shapes
:  
~
training/Adam/mul_93MulAdam/beta_2/readtraining/Adam/Variable_42/read*
T0*&
_output_shapes
:  
[
training/Adam/sub_57/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_57Subtraining/Adam/sub_57/xAdam/beta_2/read*
T0*
_output_shapes
: 
°
training/Adam/Square_18SquareMtraining/Adam/gradients/up_level_0_no_2/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:  
{
training/Adam/mul_94Multraining/Adam/sub_57training/Adam/Square_18*
T0*&
_output_shapes
:  
x
training/Adam/add_56Addtraining/Adam/mul_93training/Adam/mul_94*
T0*&
_output_shapes
:  
u
training/Adam/mul_95Multraining/Adam/multraining/Adam/add_55*
T0*&
_output_shapes
:  
[
training/Adam/Const_38Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_39Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Р
&training/Adam/clip_by_value_19/MinimumMinimumtraining/Adam/add_56training/Adam/Const_39*
T0*&
_output_shapes
:  
Ъ
training/Adam/clip_by_value_19Maximum&training/Adam/clip_by_value_19/Minimumtraining/Adam/Const_38*
T0*&
_output_shapes
:  
n
training/Adam/Sqrt_19Sqrttraining/Adam/clip_by_value_19*
T0*&
_output_shapes
:  
[
training/Adam/add_57/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
{
training/Adam/add_57Addtraining/Adam/Sqrt_19training/Adam/add_57/y*
T0*&
_output_shapes
:  
А
training/Adam/truediv_19RealDivtraining/Adam/mul_95training/Adam/add_57*
T0*&
_output_shapes
:  
Г
training/Adam/sub_58Subup_level_0_no_2/kernel/readtraining/Adam/truediv_19*
T0*&
_output_shapes
:  
Џ
training/Adam/Assign_54Assigntraining/Adam/Variable_18training/Adam/add_55*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*&
_output_shapes
:  *
use_locking(
Џ
training/Adam/Assign_55Assigntraining/Adam/Variable_42training/Adam/add_56*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_42
‘
training/Adam/Assign_56Assignup_level_0_no_2/kerneltraining/Adam/sub_58*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*)
_class
loc:@up_level_0_no_2/kernel
r
training/Adam/mul_96MulAdam/beta_1/readtraining/Adam/Variable_19/read*
T0*
_output_shapes
: 
[
training/Adam/sub_59/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_59Subtraining/Adam/sub_59/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ш
training/Adam/mul_97Multraining/Adam/sub_59@training/Adam/gradients/up_level_0_no_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
l
training/Adam/add_58Addtraining/Adam/mul_96training/Adam/mul_97*
T0*
_output_shapes
: 
r
training/Adam/mul_98MulAdam/beta_2/readtraining/Adam/Variable_43/read*
_output_shapes
: *
T0
[
training/Adam/sub_60/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_60Subtraining/Adam/sub_60/xAdam/beta_2/read*
T0*
_output_shapes
: 
И
training/Adam/Square_19Square@training/Adam/gradients/up_level_0_no_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
: 
o
training/Adam/mul_99Multraining/Adam/sub_60training/Adam/Square_19*
_output_shapes
: *
T0
l
training/Adam/add_59Addtraining/Adam/mul_98training/Adam/mul_99*
T0*
_output_shapes
: 
j
training/Adam/mul_100Multraining/Adam/multraining/Adam/add_58*
_output_shapes
: *
T0
[
training/Adam/Const_40Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_41Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Д
&training/Adam/clip_by_value_20/MinimumMinimumtraining/Adam/add_59training/Adam/Const_41*
T0*
_output_shapes
: 
О
training/Adam/clip_by_value_20Maximum&training/Adam/clip_by_value_20/Minimumtraining/Adam/Const_40*
T0*
_output_shapes
: 
b
training/Adam/Sqrt_20Sqrttraining/Adam/clip_by_value_20*
_output_shapes
: *
T0
[
training/Adam/add_60/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
o
training/Adam/add_60Addtraining/Adam/Sqrt_20training/Adam/add_60/y*
T0*
_output_shapes
: 
u
training/Adam/truediv_20RealDivtraining/Adam/mul_100training/Adam/add_60*
T0*
_output_shapes
: 
u
training/Adam/sub_61Subup_level_0_no_2/bias/readtraining/Adam/truediv_20*
T0*
_output_shapes
: 
ќ
training/Adam/Assign_57Assigntraining/Adam/Variable_19training/Adam/add_58*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19
ќ
training/Adam/Assign_58Assigntraining/Adam/Variable_43training/Adam/add_59*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_43*
validate_shape(*
_output_shapes
: 
ƒ
training/Adam/Assign_59Assignup_level_0_no_2/biastraining/Adam/sub_61*
use_locking(*
T0*'
_class
loc:@up_level_0_no_2/bias*
validate_shape(*
_output_shapes
: 

training/Adam/mul_101MulAdam/beta_1/readtraining/Adam/Variable_20/read*&
_output_shapes
: *
T0
[
training/Adam/sub_62/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_62Subtraining/Adam/sub_62/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ђ
training/Adam/mul_102Multraining/Adam/sub_62Ftraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
z
training/Adam/add_61Addtraining/Adam/mul_101training/Adam/mul_102*
T0*&
_output_shapes
: 

training/Adam/mul_103MulAdam/beta_2/readtraining/Adam/Variable_44/read*
T0*&
_output_shapes
: 
[
training/Adam/sub_63/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_63Subtraining/Adam/sub_63/xAdam/beta_2/read*
_output_shapes
: *
T0
Ъ
training/Adam/Square_20SquareFtraining/Adam/gradients/conv2d_1/convolution_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
: 
|
training/Adam/mul_104Multraining/Adam/sub_63training/Adam/Square_20*
T0*&
_output_shapes
: 
z
training/Adam/add_62Addtraining/Adam/mul_103training/Adam/mul_104*
T0*&
_output_shapes
: 
v
training/Adam/mul_105Multraining/Adam/multraining/Adam/add_61*&
_output_shapes
: *
T0
[
training/Adam/Const_42Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_43Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Р
&training/Adam/clip_by_value_21/MinimumMinimumtraining/Adam/add_62training/Adam/Const_43*
T0*&
_output_shapes
: 
Ъ
training/Adam/clip_by_value_21Maximum&training/Adam/clip_by_value_21/Minimumtraining/Adam/Const_42*
T0*&
_output_shapes
: 
n
training/Adam/Sqrt_21Sqrttraining/Adam/clip_by_value_21*
T0*&
_output_shapes
: 
[
training/Adam/add_63/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
{
training/Adam/add_63Addtraining/Adam/Sqrt_21training/Adam/add_63/y*
T0*&
_output_shapes
: 
Б
training/Adam/truediv_21RealDivtraining/Adam/mul_105training/Adam/add_63*
T0*&
_output_shapes
: 
|
training/Adam/sub_64Subconv2d_1/kernel/readtraining/Adam/truediv_21*
T0*&
_output_shapes
: 
Џ
training/Adam/Assign_60Assigntraining/Adam/Variable_20training/Adam/add_61*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_20
Џ
training/Adam/Assign_61Assigntraining/Adam/Variable_44training/Adam/add_62*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_44*
validate_shape(*&
_output_shapes
: 
∆
training/Adam/Assign_62Assignconv2d_1/kerneltraining/Adam/sub_64*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
: 
s
training/Adam/mul_106MulAdam/beta_1/readtraining/Adam/Variable_21/read*
_output_shapes
:*
T0
[
training/Adam/sub_65/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_65Subtraining/Adam/sub_65/xAdam/beta_1/read*
T0*
_output_shapes
: 
Т
training/Adam/mul_107Multraining/Adam/sub_659training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/add_64Addtraining/Adam/mul_106training/Adam/mul_107*
T0*
_output_shapes
:
s
training/Adam/mul_108MulAdam/beta_2/readtraining/Adam/Variable_45/read*
T0*
_output_shapes
:
[
training/Adam/sub_66/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_66Subtraining/Adam/sub_66/xAdam/beta_2/read*
_output_shapes
: *
T0
Б
training/Adam/Square_21Square9training/Adam/gradients/conv2d_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
p
training/Adam/mul_109Multraining/Adam/sub_66training/Adam/Square_21*
_output_shapes
:*
T0
n
training/Adam/add_65Addtraining/Adam/mul_108training/Adam/mul_109*
T0*
_output_shapes
:
j
training/Adam/mul_110Multraining/Adam/multraining/Adam/add_64*
_output_shapes
:*
T0
[
training/Adam/Const_44Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_45Const*
dtype0*
_output_shapes
: *
valueB
 *  А
Д
&training/Adam/clip_by_value_22/MinimumMinimumtraining/Adam/add_65training/Adam/Const_45*
T0*
_output_shapes
:
О
training/Adam/clip_by_value_22Maximum&training/Adam/clip_by_value_22/Minimumtraining/Adam/Const_44*
T0*
_output_shapes
:
b
training/Adam/Sqrt_22Sqrttraining/Adam/clip_by_value_22*
T0*
_output_shapes
:
[
training/Adam/add_66/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
o
training/Adam/add_66Addtraining/Adam/Sqrt_22training/Adam/add_66/y*
T0*
_output_shapes
:
u
training/Adam/truediv_22RealDivtraining/Adam/mul_110training/Adam/add_66*
_output_shapes
:*
T0
n
training/Adam/sub_67Subconv2d_1/bias/readtraining/Adam/truediv_22*
T0*
_output_shapes
:
ќ
training/Adam/Assign_63Assigntraining/Adam/Variable_21training/Adam/add_64*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes
:*
use_locking(
ќ
training/Adam/Assign_64Assigntraining/Adam/Variable_45training/Adam/add_65*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_45
ґ
training/Adam/Assign_65Assignconv2d_1/biastraining/Adam/sub_67*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/mul_111MulAdam/beta_1/readtraining/Adam/Variable_22/read*&
_output_shapes
: *
T0
[
training/Adam/sub_68/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_68Subtraining/Adam/sub_68/xAdam/beta_1/read*
T0*
_output_shapes
: 
Ђ
training/Adam/mul_112Multraining/Adam/sub_68Ftraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: *
T0
z
training/Adam/add_67Addtraining/Adam/mul_111training/Adam/mul_112*
T0*&
_output_shapes
: 

training/Adam/mul_113MulAdam/beta_2/readtraining/Adam/Variable_46/read*
T0*&
_output_shapes
: 
[
training/Adam/sub_69/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_69Subtraining/Adam/sub_69/xAdam/beta_2/read*
T0*
_output_shapes
: 
Ъ
training/Adam/Square_22SquareFtraining/Adam/gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: *
T0
|
training/Adam/mul_114Multraining/Adam/sub_69training/Adam/Square_22*
T0*&
_output_shapes
: 
z
training/Adam/add_68Addtraining/Adam/mul_113training/Adam/mul_114*&
_output_shapes
: *
T0
v
training/Adam/mul_115Multraining/Adam/multraining/Adam/add_67*&
_output_shapes
: *
T0
[
training/Adam/Const_46Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_47Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Р
&training/Adam/clip_by_value_23/MinimumMinimumtraining/Adam/add_68training/Adam/Const_47*
T0*&
_output_shapes
: 
Ъ
training/Adam/clip_by_value_23Maximum&training/Adam/clip_by_value_23/Minimumtraining/Adam/Const_46*&
_output_shapes
: *
T0
n
training/Adam/Sqrt_23Sqrttraining/Adam/clip_by_value_23*
T0*&
_output_shapes
: 
[
training/Adam/add_69/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
{
training/Adam/add_69Addtraining/Adam/Sqrt_23training/Adam/add_69/y*
T0*&
_output_shapes
: 
Б
training/Adam/truediv_23RealDivtraining/Adam/mul_115training/Adam/add_69*&
_output_shapes
: *
T0
|
training/Adam/sub_70Subconv2d_2/kernel/readtraining/Adam/truediv_23*
T0*&
_output_shapes
: 
Џ
training/Adam/Assign_66Assigntraining/Adam/Variable_22training/Adam/add_67*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*&
_output_shapes
: *
use_locking(
Џ
training/Adam/Assign_67Assigntraining/Adam/Variable_46training/Adam/add_68*
T0*,
_class"
 loc:@training/Adam/Variable_46*
validate_shape(*&
_output_shapes
: *
use_locking(
∆
training/Adam/Assign_68Assignconv2d_2/kerneltraining/Adam/sub_70*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(
s
training/Adam/mul_116MulAdam/beta_1/readtraining/Adam/Variable_23/read*
_output_shapes
:*
T0
[
training/Adam/sub_71/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_71Subtraining/Adam/sub_71/xAdam/beta_1/read*
T0*
_output_shapes
: 
Т
training/Adam/mul_117Multraining/Adam/sub_719training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/add_70Addtraining/Adam/mul_116training/Adam/mul_117*
T0*
_output_shapes
:
s
training/Adam/mul_118MulAdam/beta_2/readtraining/Adam/Variable_47/read*
_output_shapes
:*
T0
[
training/Adam/sub_72/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_72Subtraining/Adam/sub_72/xAdam/beta_2/read*
T0*
_output_shapes
: 
Б
training/Adam/Square_23Square9training/Adam/gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
p
training/Adam/mul_119Multraining/Adam/sub_72training/Adam/Square_23*
T0*
_output_shapes
:
n
training/Adam/add_71Addtraining/Adam/mul_118training/Adam/mul_119*
T0*
_output_shapes
:
j
training/Adam/mul_120Multraining/Adam/multraining/Adam/add_70*
T0*
_output_shapes
:
[
training/Adam/Const_48Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_49Const*
valueB
 *  А*
dtype0*
_output_shapes
: 
Д
&training/Adam/clip_by_value_24/MinimumMinimumtraining/Adam/add_71training/Adam/Const_49*
T0*
_output_shapes
:
О
training/Adam/clip_by_value_24Maximum&training/Adam/clip_by_value_24/Minimumtraining/Adam/Const_48*
T0*
_output_shapes
:
b
training/Adam/Sqrt_24Sqrttraining/Adam/clip_by_value_24*
T0*
_output_shapes
:
[
training/Adam/add_72/yConst*
dtype0*
_output_shapes
: *
valueB
 *Хњ÷3
o
training/Adam/add_72Addtraining/Adam/Sqrt_24training/Adam/add_72/y*
_output_shapes
:*
T0
u
training/Adam/truediv_24RealDivtraining/Adam/mul_120training/Adam/add_72*
T0*
_output_shapes
:
n
training/Adam/sub_73Subconv2d_2/bias/readtraining/Adam/truediv_24*
T0*
_output_shapes
:
ќ
training/Adam/Assign_69Assigntraining/Adam/Variable_23training/Adam/add_70*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23
ќ
training/Adam/Assign_70Assigntraining/Adam/Variable_47training/Adam/add_71*
T0*,
_class"
 loc:@training/Adam/Variable_47*
validate_shape(*
_output_shapes
:*
use_locking(
ґ
training/Adam/Assign_71Assignconv2d_2/biastraining/Adam/sub_73*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@conv2d_2/bias
Ѓ
training/group_depsNoOp	^loss/mul^metrics/mae/Mean_1^metrics/mse/Mean_1^training/Adam/Assign^training/Adam/AssignAdd^training/Adam/Assign_1^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_2^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23^training/Adam/Assign_24^training/Adam/Assign_25^training/Adam/Assign_26^training/Adam/Assign_27^training/Adam/Assign_28^training/Adam/Assign_29^training/Adam/Assign_3^training/Adam/Assign_30^training/Adam/Assign_31^training/Adam/Assign_32^training/Adam/Assign_33^training/Adam/Assign_34^training/Adam/Assign_35^training/Adam/Assign_36^training/Adam/Assign_37^training/Adam/Assign_38^training/Adam/Assign_39^training/Adam/Assign_4^training/Adam/Assign_40^training/Adam/Assign_41^training/Adam/Assign_42^training/Adam/Assign_43^training/Adam/Assign_44^training/Adam/Assign_45^training/Adam/Assign_46^training/Adam/Assign_47^training/Adam/Assign_48^training/Adam/Assign_49^training/Adam/Assign_5^training/Adam/Assign_50^training/Adam/Assign_51^training/Adam/Assign_52^training/Adam/Assign_53^training/Adam/Assign_54^training/Adam/Assign_55^training/Adam/Assign_56^training/Adam/Assign_57^training/Adam/Assign_58^training/Adam/Assign_59^training/Adam/Assign_6^training/Adam/Assign_60^training/Adam/Assign_61^training/Adam/Assign_62^training/Adam/Assign_63^training/Adam/Assign_64^training/Adam/Assign_65^training/Adam/Assign_66^training/Adam/Assign_67^training/Adam/Assign_68^training/Adam/Assign_69^training/Adam/Assign_7^training/Adam/Assign_70^training/Adam/Assign_71^training/Adam/Assign_8^training/Adam/Assign_9
G

group_depsNoOp	^loss/mul^metrics/mae/Mean_1^metrics/mse/Mean_1
Ъ
IsVariableInitializedIsVariableInitializeddown_level_0_no_0/kernel*
dtype0*
_output_shapes
: *+
_class!
loc:@down_level_0_no_0/kernel
Ш
IsVariableInitialized_1IsVariableInitializeddown_level_0_no_0/bias*)
_class
loc:@down_level_0_no_0/bias*
dtype0*
_output_shapes
: 
Ь
IsVariableInitialized_2IsVariableInitializeddown_level_0_no_1/kernel*+
_class!
loc:@down_level_0_no_1/kernel*
dtype0*
_output_shapes
: 
Ш
IsVariableInitialized_3IsVariableInitializeddown_level_0_no_1/bias*)
_class
loc:@down_level_0_no_1/bias*
dtype0*
_output_shapes
: 
Ь
IsVariableInitialized_4IsVariableInitializeddown_level_1_no_0/kernel*+
_class!
loc:@down_level_1_no_0/kernel*
dtype0*
_output_shapes
: 
Ш
IsVariableInitialized_5IsVariableInitializeddown_level_1_no_0/bias*)
_class
loc:@down_level_1_no_0/bias*
dtype0*
_output_shapes
: 
Ь
IsVariableInitialized_6IsVariableInitializeddown_level_1_no_1/kernel*+
_class!
loc:@down_level_1_no_1/kernel*
dtype0*
_output_shapes
: 
Ш
IsVariableInitialized_7IsVariableInitializeddown_level_1_no_1/bias*)
_class
loc:@down_level_1_no_1/bias*
dtype0*
_output_shapes
: 
К
IsVariableInitialized_8IsVariableInitializedmiddle_0/kernel*"
_class
loc:@middle_0/kernel*
dtype0*
_output_shapes
: 
Ж
IsVariableInitialized_9IsVariableInitializedmiddle_0/bias* 
_class
loc:@middle_0/bias*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_10IsVariableInitializedmiddle_2/kernel*"
_class
loc:@middle_2/kernel*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_11IsVariableInitializedmiddle_2/bias* 
_class
loc:@middle_2/bias*
dtype0*
_output_shapes
: 
Щ
IsVariableInitialized_12IsVariableInitializedup_level_1_no_0/kernel*
dtype0*
_output_shapes
: *)
_class
loc:@up_level_1_no_0/kernel
Х
IsVariableInitialized_13IsVariableInitializedup_level_1_no_0/bias*'
_class
loc:@up_level_1_no_0/bias*
dtype0*
_output_shapes
: 
Щ
IsVariableInitialized_14IsVariableInitializedup_level_1_no_2/kernel*)
_class
loc:@up_level_1_no_2/kernel*
dtype0*
_output_shapes
: 
Х
IsVariableInitialized_15IsVariableInitializedup_level_1_no_2/bias*
dtype0*
_output_shapes
: *'
_class
loc:@up_level_1_no_2/bias
Щ
IsVariableInitialized_16IsVariableInitializedup_level_0_no_0/kernel*)
_class
loc:@up_level_0_no_0/kernel*
dtype0*
_output_shapes
: 
Х
IsVariableInitialized_17IsVariableInitializedup_level_0_no_0/bias*
dtype0*
_output_shapes
: *'
_class
loc:@up_level_0_no_0/bias
Щ
IsVariableInitialized_18IsVariableInitializedup_level_0_no_2/kernel*
dtype0*
_output_shapes
: *)
_class
loc:@up_level_0_no_2/kernel
Х
IsVariableInitialized_19IsVariableInitializedup_level_0_no_2/bias*'
_class
loc:@up_level_0_no_2/bias*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_20IsVariableInitializedconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_21IsVariableInitializedconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_22IsVariableInitializedconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_23IsVariableInitializedconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_24IsVariableInitializedAdam/iterations*
dtype0	*
_output_shapes
: *"
_class
loc:@Adam/iterations
{
IsVariableInitialized_25IsVariableInitializedAdam/lr*
_class
loc:@Adam/lr*
dtype0*
_output_shapes
: 
Г
IsVariableInitialized_26IsVariableInitializedAdam/beta_1*
_class
loc:@Adam/beta_1*
dtype0*
_output_shapes
: 
Г
IsVariableInitialized_27IsVariableInitializedAdam/beta_2*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_2
Б
IsVariableInitialized_28IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 
Щ
IsVariableInitialized_29IsVariableInitializedtraining/Adam/Variable*
dtype0*
_output_shapes
: *)
_class
loc:@training/Adam/Variable
Э
IsVariableInitialized_30IsVariableInitializedtraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
dtype0*
_output_shapes
: 
Э
IsVariableInitialized_31IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
dtype0*
_output_shapes
: 
Э
IsVariableInitialized_32IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
dtype0*
_output_shapes
: 
Э
IsVariableInitialized_33IsVariableInitializedtraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0*
_output_shapes
: 
Э
IsVariableInitialized_34IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 
Э
IsVariableInitialized_35IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
dtype0*
_output_shapes
: 
Э
IsVariableInitialized_36IsVariableInitializedtraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
dtype0*
_output_shapes
: 
Э
IsVariableInitialized_37IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
dtype0*
_output_shapes
: 
Э
IsVariableInitialized_38IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_39IsVariableInitializedtraining/Adam/Variable_10*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_10
Я
IsVariableInitialized_40IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_41IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_42IsVariableInitializedtraining/Adam/Variable_13*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_13
Я
IsVariableInitialized_43IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_44IsVariableInitializedtraining/Adam/Variable_15*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_15
Я
IsVariableInitialized_45IsVariableInitializedtraining/Adam/Variable_16*,
_class"
 loc:@training/Adam/Variable_16*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_46IsVariableInitializedtraining/Adam/Variable_17*,
_class"
 loc:@training/Adam/Variable_17*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_47IsVariableInitializedtraining/Adam/Variable_18*,
_class"
 loc:@training/Adam/Variable_18*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_48IsVariableInitializedtraining/Adam/Variable_19*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_19
Я
IsVariableInitialized_49IsVariableInitializedtraining/Adam/Variable_20*,
_class"
 loc:@training/Adam/Variable_20*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_50IsVariableInitializedtraining/Adam/Variable_21*,
_class"
 loc:@training/Adam/Variable_21*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_51IsVariableInitializedtraining/Adam/Variable_22*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_22
Я
IsVariableInitialized_52IsVariableInitializedtraining/Adam/Variable_23*,
_class"
 loc:@training/Adam/Variable_23*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_53IsVariableInitializedtraining/Adam/Variable_24*,
_class"
 loc:@training/Adam/Variable_24*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_54IsVariableInitializedtraining/Adam/Variable_25*,
_class"
 loc:@training/Adam/Variable_25*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_55IsVariableInitializedtraining/Adam/Variable_26*,
_class"
 loc:@training/Adam/Variable_26*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_56IsVariableInitializedtraining/Adam/Variable_27*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_27
Я
IsVariableInitialized_57IsVariableInitializedtraining/Adam/Variable_28*,
_class"
 loc:@training/Adam/Variable_28*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_58IsVariableInitializedtraining/Adam/Variable_29*,
_class"
 loc:@training/Adam/Variable_29*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_59IsVariableInitializedtraining/Adam/Variable_30*,
_class"
 loc:@training/Adam/Variable_30*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_60IsVariableInitializedtraining/Adam/Variable_31*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_31
Я
IsVariableInitialized_61IsVariableInitializedtraining/Adam/Variable_32*,
_class"
 loc:@training/Adam/Variable_32*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_62IsVariableInitializedtraining/Adam/Variable_33*,
_class"
 loc:@training/Adam/Variable_33*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_63IsVariableInitializedtraining/Adam/Variable_34*,
_class"
 loc:@training/Adam/Variable_34*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_64IsVariableInitializedtraining/Adam/Variable_35*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_35
Я
IsVariableInitialized_65IsVariableInitializedtraining/Adam/Variable_36*,
_class"
 loc:@training/Adam/Variable_36*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_66IsVariableInitializedtraining/Adam/Variable_37*,
_class"
 loc:@training/Adam/Variable_37*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_67IsVariableInitializedtraining/Adam/Variable_38*,
_class"
 loc:@training/Adam/Variable_38*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_68IsVariableInitializedtraining/Adam/Variable_39*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_39
Я
IsVariableInitialized_69IsVariableInitializedtraining/Adam/Variable_40*,
_class"
 loc:@training/Adam/Variable_40*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_70IsVariableInitializedtraining/Adam/Variable_41*,
_class"
 loc:@training/Adam/Variable_41*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_71IsVariableInitializedtraining/Adam/Variable_42*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_42
Я
IsVariableInitialized_72IsVariableInitializedtraining/Adam/Variable_43*,
_class"
 loc:@training/Adam/Variable_43*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_73IsVariableInitializedtraining/Adam/Variable_44*,
_class"
 loc:@training/Adam/Variable_44*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_74IsVariableInitializedtraining/Adam/Variable_45*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_45
Я
IsVariableInitialized_75IsVariableInitializedtraining/Adam/Variable_46*,
_class"
 loc:@training/Adam/Variable_46*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_76IsVariableInitializedtraining/Adam/Variable_47*,
_class"
 loc:@training/Adam/Variable_47*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_77IsVariableInitializedtraining/Adam/Variable_48*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_48
Я
IsVariableInitialized_78IsVariableInitializedtraining/Adam/Variable_49*,
_class"
 loc:@training/Adam/Variable_49*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_79IsVariableInitializedtraining/Adam/Variable_50*,
_class"
 loc:@training/Adam/Variable_50*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_80IsVariableInitializedtraining/Adam/Variable_51*,
_class"
 loc:@training/Adam/Variable_51*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_81IsVariableInitializedtraining/Adam/Variable_52*,
_class"
 loc:@training/Adam/Variable_52*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_82IsVariableInitializedtraining/Adam/Variable_53*,
_class"
 loc:@training/Adam/Variable_53*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_83IsVariableInitializedtraining/Adam/Variable_54*,
_class"
 loc:@training/Adam/Variable_54*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_84IsVariableInitializedtraining/Adam/Variable_55*,
_class"
 loc:@training/Adam/Variable_55*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_85IsVariableInitializedtraining/Adam/Variable_56*,
_class"
 loc:@training/Adam/Variable_56*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_86IsVariableInitializedtraining/Adam/Variable_57*,
_class"
 loc:@training/Adam/Variable_57*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_87IsVariableInitializedtraining/Adam/Variable_58*,
_class"
 loc:@training/Adam/Variable_58*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_88IsVariableInitializedtraining/Adam/Variable_59*,
_class"
 loc:@training/Adam/Variable_59*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_89IsVariableInitializedtraining/Adam/Variable_60*,
_class"
 loc:@training/Adam/Variable_60*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_90IsVariableInitializedtraining/Adam/Variable_61*,
_class"
 loc:@training/Adam/Variable_61*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_91IsVariableInitializedtraining/Adam/Variable_62*,
_class"
 loc:@training/Adam/Variable_62*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_92IsVariableInitializedtraining/Adam/Variable_63*,
_class"
 loc:@training/Adam/Variable_63*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_93IsVariableInitializedtraining/Adam/Variable_64*,
_class"
 loc:@training/Adam/Variable_64*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_94IsVariableInitializedtraining/Adam/Variable_65*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_65
Я
IsVariableInitialized_95IsVariableInitializedtraining/Adam/Variable_66*,
_class"
 loc:@training/Adam/Variable_66*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_96IsVariableInitializedtraining/Adam/Variable_67*,
_class"
 loc:@training/Adam/Variable_67*
dtype0*
_output_shapes
: 
Я
IsVariableInitialized_97IsVariableInitializedtraining/Adam/Variable_68*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_68
Я
IsVariableInitialized_98IsVariableInitializedtraining/Adam/Variable_69*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_69
Я
IsVariableInitialized_99IsVariableInitializedtraining/Adam/Variable_70*,
_class"
 loc:@training/Adam/Variable_70*
dtype0*
_output_shapes
: 
†
IsVariableInitialized_100IsVariableInitializedtraining/Adam/Variable_71*,
_class"
 loc:@training/Adam/Variable_71*
dtype0*
_output_shapes
: 
А
initNoOp^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^Adam/iterations/Assign^Adam/lr/Assign^conv2d_1/bias/Assign^conv2d_1/kernel/Assign^conv2d_2/bias/Assign^conv2d_2/kernel/Assign^down_level_0_no_0/bias/Assign ^down_level_0_no_0/kernel/Assign^down_level_0_no_1/bias/Assign ^down_level_0_no_1/kernel/Assign^down_level_1_no_0/bias/Assign ^down_level_1_no_0/kernel/Assign^down_level_1_no_1/bias/Assign ^down_level_1_no_1/kernel/Assign^middle_0/bias/Assign^middle_0/kernel/Assign^middle_2/bias/Assign^middle_2/kernel/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign!^training/Adam/Variable_16/Assign!^training/Adam/Variable_17/Assign!^training/Adam/Variable_18/Assign!^training/Adam/Variable_19/Assign ^training/Adam/Variable_2/Assign!^training/Adam/Variable_20/Assign!^training/Adam/Variable_21/Assign!^training/Adam/Variable_22/Assign!^training/Adam/Variable_23/Assign!^training/Adam/Variable_24/Assign!^training/Adam/Variable_25/Assign!^training/Adam/Variable_26/Assign!^training/Adam/Variable_27/Assign!^training/Adam/Variable_28/Assign!^training/Adam/Variable_29/Assign ^training/Adam/Variable_3/Assign!^training/Adam/Variable_30/Assign!^training/Adam/Variable_31/Assign!^training/Adam/Variable_32/Assign!^training/Adam/Variable_33/Assign!^training/Adam/Variable_34/Assign!^training/Adam/Variable_35/Assign!^training/Adam/Variable_36/Assign!^training/Adam/Variable_37/Assign!^training/Adam/Variable_38/Assign!^training/Adam/Variable_39/Assign ^training/Adam/Variable_4/Assign!^training/Adam/Variable_40/Assign!^training/Adam/Variable_41/Assign!^training/Adam/Variable_42/Assign!^training/Adam/Variable_43/Assign!^training/Adam/Variable_44/Assign!^training/Adam/Variable_45/Assign!^training/Adam/Variable_46/Assign!^training/Adam/Variable_47/Assign!^training/Adam/Variable_48/Assign!^training/Adam/Variable_49/Assign ^training/Adam/Variable_5/Assign!^training/Adam/Variable_50/Assign!^training/Adam/Variable_51/Assign!^training/Adam/Variable_52/Assign!^training/Adam/Variable_53/Assign!^training/Adam/Variable_54/Assign!^training/Adam/Variable_55/Assign!^training/Adam/Variable_56/Assign!^training/Adam/Variable_57/Assign!^training/Adam/Variable_58/Assign!^training/Adam/Variable_59/Assign ^training/Adam/Variable_6/Assign!^training/Adam/Variable_60/Assign!^training/Adam/Variable_61/Assign!^training/Adam/Variable_62/Assign!^training/Adam/Variable_63/Assign!^training/Adam/Variable_64/Assign!^training/Adam/Variable_65/Assign!^training/Adam/Variable_66/Assign!^training/Adam/Variable_67/Assign!^training/Adam/Variable_68/Assign!^training/Adam/Variable_69/Assign ^training/Adam/Variable_7/Assign!^training/Adam/Variable_70/Assign!^training/Adam/Variable_71/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign^up_level_0_no_0/bias/Assign^up_level_0_no_0/kernel/Assign^up_level_0_no_2/bias/Assign^up_level_0_no_2/kernel/Assign^up_level_1_no_0/bias/Assign^up_level_1_no_0/kernel/Assign^up_level_1_no_2/bias/Assign^up_level_1_no_2/kernel/Assign
Ґ
PlaceholderPlaceholder*6
shape-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
dtype0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
§
strided_sliceStridedSliceinputstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Y
lambda_2/percentile/q/xConst*
value	B :*
dtype0*
_output_shapes
: 
v
lambda_2/percentile/qCastlambda_2/percentile/q/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
i
lambda_2/percentile/axisConst*
valueB"      *
dtype0*
_output_shapes
:
L
Dlambda_2/percentile/assert_integer/statically_determined_was_integerNoOp
{
"lambda_2/percentile/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ґ
lambda_2/percentile/transpose	Transposestrided_slice"lambda_2/percentile/transpose/perm*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0
v
lambda_2/percentile/ShapeShapelambda_2/percentile/transpose*
T0*
out_type0*
_output_shapes
:
q
'lambda_2/percentile/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
s
)lambda_2/percentile/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)lambda_2/percentile/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
б
!lambda_2/percentile/strided_sliceStridedSlicelambda_2/percentile/Shape'lambda_2/percentile/strided_slice/stack)lambda_2/percentile/strided_slice/stack_1)lambda_2/percentile/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
v
#lambda_2/percentile/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
a
lambda_2/percentile/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
…
lambda_2/percentile/concatConcatV2!lambda_2/percentile/strided_slice#lambda_2/percentile/concat/values_1lambda_2/percentile/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
Ѓ
lambda_2/percentile/ReshapeReshapelambda_2/percentile/transposelambda_2/percentile/concat*
T0*
Tshape0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
f
lambda_2/percentile/truediv/yConst*
dtype0*
_output_shapes
: *
valueB 2      Y@
}
lambda_2/percentile/truedivRealDivlambda_2/percentile/qlambda_2/percentile/truediv/y*
T0*
_output_shapes
: 
b
lambda_2/percentile/sub/xConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
w
lambda_2/percentile/subSublambda_2/percentile/sub/xlambda_2/percentile/truediv*
T0*
_output_shapes
: 
v
lambda_2/percentile/Shape_1Shapelambda_2/percentile/Reshape*
T0*
out_type0*
_output_shapes
:
|
)lambda_2/percentile/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
u
+lambda_2/percentile/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
u
+lambda_2/percentile/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
з
#lambda_2/percentile/strided_slice_1StridedSlicelambda_2/percentile/Shape_1)lambda_2/percentile/strided_slice_1/stack+lambda_2/percentile/strided_slice_1/stack_1+lambda_2/percentile/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
Й
lambda_2/percentile/ToDoubleCast#lambda_2/percentile/strided_slice_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
d
lambda_2/percentile/sub_1/yConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
|
lambda_2/percentile/sub_1Sublambda_2/percentile/ToDoublelambda_2/percentile/sub_1/y*
T0*
_output_shapes
: 
s
lambda_2/percentile/mulMullambda_2/percentile/sub_1lambda_2/percentile/sub*
T0*
_output_shapes
: 
\
lambda_2/percentile/RoundRoundlambda_2/percentile/mul*
T0*
_output_shapes
: 
v
lambda_2/percentile/Shape_2Shapelambda_2/percentile/Reshape*
T0*
out_type0*
_output_shapes
:
|
)lambda_2/percentile/strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
u
+lambda_2/percentile/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_2/percentile/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
з
#lambda_2/percentile/strided_slice_2StridedSlicelambda_2/percentile/Shape_2)lambda_2/percentile/strided_slice_2/stack+lambda_2/percentile/strided_slice_2/stack_1+lambda_2/percentile/strided_slice_2/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
~
lambda_2/percentile/ToInt32Castlambda_2/percentile/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
]
lambda_2/percentile/sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Г
lambda_2/percentile/sub_2Sub#lambda_2/percentile/strided_slice_2lambda_2/percentile/sub_2/y*
_output_shapes
: *
T0
Н
)lambda_2/percentile/clip_by_value/MinimumMinimumlambda_2/percentile/ToInt32lambda_2/percentile/sub_2*
T0*
_output_shapes
: 
e
#lambda_2/percentile/clip_by_value/yConst*
dtype0*
_output_shapes
: *
value	B : 
Э
!lambda_2/percentile/clip_by_valueMaximum)lambda_2/percentile/clip_by_value/Minimum#lambda_2/percentile/clip_by_value/y*
T0*
_output_shapes
: 
v
lambda_2/percentile/Shape_3Shapelambda_2/percentile/Reshape*
T0*
out_type0*
_output_shapes
:
|
)lambda_2/percentile/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
u
+lambda_2/percentile/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_2/percentile/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
з
#lambda_2/percentile/strided_slice_3StridedSlicelambda_2/percentile/Shape_3)lambda_2/percentile/strided_slice_3/stack+lambda_2/percentile/strided_slice_3/stack_1+lambda_2/percentile/strided_slice_3/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
”
lambda_2/percentile/TopKV2TopKV2lambda_2/percentile/Reshape#lambda_2/percentile/strided_slice_3*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€*
sorted(
[
lambda_2/percentile/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
}
lambda_2/percentile/addAdd!lambda_2/percentile/clip_by_valuelambda_2/percentile/add/y*
T0*
_output_shapes
: 
m
+lambda_2/percentile/strided_slice_4/stack/0Const*
dtype0*
_output_shapes
: *
value	B : 
ї
)lambda_2/percentile/strided_slice_4/stackPack+lambda_2/percentile/strided_slice_4/stack/0!lambda_2/percentile/clip_by_value*
T0*

axis *
N*
_output_shapes
:
o
-lambda_2/percentile/strided_slice_4/stack_1/0Const*
dtype0*
_output_shapes
: *
value	B : 
µ
+lambda_2/percentile/strided_slice_4/stack_1Pack-lambda_2/percentile/strided_slice_4/stack_1/0lambda_2/percentile/add*
T0*

axis *
N*
_output_shapes
:
|
+lambda_2/percentile/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
ч
#lambda_2/percentile/strided_slice_4StridedSlicelambda_2/percentile/TopKV2)lambda_2/percentile/strided_slice_4/stack+lambda_2/percentile/strided_slice_4/stack_1+lambda_2/percentile/strided_slice_4/stack_2*
ellipsis_mask*

begin_mask *
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€*
T0*
Index0*
shrink_axis_mask
d
"lambda_2/percentile/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ј
lambda_2/percentile/ExpandDims
ExpandDims#lambda_2/percentile/strided_slice_4"lambda_2/percentile/ExpandDims/dim*+
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
f
$lambda_2/percentile/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ї
 lambda_2/percentile/ExpandDims_1
ExpandDimslambda_2/percentile/ExpandDims$lambda_2/percentile/ExpandDims_1/dim*/
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
^
lambda_2/percentile_1/q/xConst*
valueB
 *ЪЩ«B*
dtype0*
_output_shapes
: 
z
lambda_2/percentile_1/qCastlambda_2/percentile_1/q/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
k
lambda_2/percentile_1/axisConst*
valueB"      *
dtype0*
_output_shapes
:
N
Flambda_2/percentile_1/assert_integer/statically_determined_was_integerNoOp
}
$lambda_2/percentile_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ї
lambda_2/percentile_1/transpose	Transposestrided_slice$lambda_2/percentile_1/transpose/perm*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0
z
lambda_2/percentile_1/ShapeShapelambda_2/percentile_1/transpose*
T0*
out_type0*
_output_shapes
:
s
)lambda_2/percentile_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_2/percentile_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+lambda_2/percentile_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
л
#lambda_2/percentile_1/strided_sliceStridedSlicelambda_2/percentile_1/Shape)lambda_2/percentile_1/strided_slice/stack+lambda_2/percentile_1/strided_slice/stack_1+lambda_2/percentile_1/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
x
%lambda_2/percentile_1/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
c
!lambda_2/percentile_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
—
lambda_2/percentile_1/concatConcatV2#lambda_2/percentile_1/strided_slice%lambda_2/percentile_1/concat/values_1!lambda_2/percentile_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
і
lambda_2/percentile_1/ReshapeReshapelambda_2/percentile_1/transposelambda_2/percentile_1/concat*
T0*
Tshape0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
h
lambda_2/percentile_1/truediv/yConst*
valueB 2      Y@*
dtype0*
_output_shapes
: 
Г
lambda_2/percentile_1/truedivRealDivlambda_2/percentile_1/qlambda_2/percentile_1/truediv/y*
T0*
_output_shapes
: 
d
lambda_2/percentile_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB 2      р?
}
lambda_2/percentile_1/subSublambda_2/percentile_1/sub/xlambda_2/percentile_1/truediv*
T0*
_output_shapes
: 
z
lambda_2/percentile_1/Shape_1Shapelambda_2/percentile_1/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_2/percentile_1/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w
-lambda_2/percentile_1/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
w
-lambda_2/percentile_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_2/percentile_1/strided_slice_1StridedSlicelambda_2/percentile_1/Shape_1+lambda_2/percentile_1/strided_slice_1/stack-lambda_2/percentile_1/strided_slice_1/stack_1-lambda_2/percentile_1/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Н
lambda_2/percentile_1/ToDoubleCast%lambda_2/percentile_1/strided_slice_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
f
lambda_2/percentile_1/sub_1/yConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
В
lambda_2/percentile_1/sub_1Sublambda_2/percentile_1/ToDoublelambda_2/percentile_1/sub_1/y*
T0*
_output_shapes
: 
y
lambda_2/percentile_1/mulMullambda_2/percentile_1/sub_1lambda_2/percentile_1/sub*
T0*
_output_shapes
: 
`
lambda_2/percentile_1/RoundRoundlambda_2/percentile_1/mul*
T0*
_output_shapes
: 
z
lambda_2/percentile_1/Shape_2Shapelambda_2/percentile_1/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_2/percentile_1/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w
-lambda_2/percentile_1/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_2/percentile_1/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_2/percentile_1/strided_slice_2StridedSlicelambda_2/percentile_1/Shape_2+lambda_2/percentile_1/strided_slice_2/stack-lambda_2/percentile_1/strided_slice_2/stack_1-lambda_2/percentile_1/strided_slice_2/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
В
lambda_2/percentile_1/ToInt32Castlambda_2/percentile_1/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
_
lambda_2/percentile_1/sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Й
lambda_2/percentile_1/sub_2Sub%lambda_2/percentile_1/strided_slice_2lambda_2/percentile_1/sub_2/y*
T0*
_output_shapes
: 
У
+lambda_2/percentile_1/clip_by_value/MinimumMinimumlambda_2/percentile_1/ToInt32lambda_2/percentile_1/sub_2*
T0*
_output_shapes
: 
g
%lambda_2/percentile_1/clip_by_value/yConst*
value	B : *
dtype0*
_output_shapes
: 
£
#lambda_2/percentile_1/clip_by_valueMaximum+lambda_2/percentile_1/clip_by_value/Minimum%lambda_2/percentile_1/clip_by_value/y*
T0*
_output_shapes
: 
z
lambda_2/percentile_1/Shape_3Shapelambda_2/percentile_1/Reshape*
_output_shapes
:*
T0*
out_type0
~
+lambda_2/percentile_1/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w
-lambda_2/percentile_1/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_2/percentile_1/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_2/percentile_1/strided_slice_3StridedSlicelambda_2/percentile_1/Shape_3+lambda_2/percentile_1/strided_slice_3/stack-lambda_2/percentile_1/strided_slice_3/stack_1-lambda_2/percentile_1/strided_slice_3/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
ў
lambda_2/percentile_1/TopKV2TopKV2lambda_2/percentile_1/Reshape%lambda_2/percentile_1/strided_slice_3*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€*
sorted(
]
lambda_2/percentile_1/add/yConst*
dtype0*
_output_shapes
: *
value	B :
Г
lambda_2/percentile_1/addAdd#lambda_2/percentile_1/clip_by_valuelambda_2/percentile_1/add/y*
T0*
_output_shapes
: 
o
-lambda_2/percentile_1/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
+lambda_2/percentile_1/strided_slice_4/stackPack-lambda_2/percentile_1/strided_slice_4/stack/0#lambda_2/percentile_1/clip_by_value*
T0*

axis *
N*
_output_shapes
:
q
/lambda_2/percentile_1/strided_slice_4/stack_1/0Const*
dtype0*
_output_shapes
: *
value	B : 
ї
-lambda_2/percentile_1/strided_slice_4/stack_1Pack/lambda_2/percentile_1/strided_slice_4/stack_1/0lambda_2/percentile_1/add*
T0*

axis *
N*
_output_shapes
:
~
-lambda_2/percentile_1/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
Б
%lambda_2/percentile_1/strided_slice_4StridedSlicelambda_2/percentile_1/TopKV2+lambda_2/percentile_1/strided_slice_4/stack-lambda_2/percentile_1/strided_slice_4/stack_1-lambda_2/percentile_1/strided_slice_4/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask*
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€*
T0*
Index0
f
$lambda_2/percentile_1/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
љ
 lambda_2/percentile_1/ExpandDims
ExpandDims%lambda_2/percentile_1/strided_slice_4$lambda_2/percentile_1/ExpandDims/dim*+
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
h
&lambda_2/percentile_1/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B :
ј
"lambda_2/percentile_1/ExpandDims_1
ExpandDims lambda_2/percentile_1/ExpandDims&lambda_2/percentile_1/ExpandDims_1/dim*

Tdim0*
T0*/
_output_shapes
:€€€€€€€€€
Р
lambda_2/subSubstrided_slice lambda_2/percentile/ExpandDims_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Х
lambda_2/sub_1Sub"lambda_2/percentile_1/ExpandDims_1 lambda_2/percentile/ExpandDims_1*/
_output_shapes
:€€€€€€€€€*
T0
S
lambda_2/add/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
m
lambda_2/addAddlambda_2/sub_1lambda_2/add/y*
T0*/
_output_shapes
:€€€€€€€€€
Г
lambda_2/truedivRealDivlambda_2/sublambda_2/add*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
S
lambda_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
U
lambda_2/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Щ
lambda_2/clip_by_value/MinimumMinimumlambda_2/truedivlambda_2/Const_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Э
lambda_2/clip_by_valueMaximumlambda_2/clip_by_value/Minimumlambda_2/Const*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ђ
lambda_2/PlaceholderPlaceholder*6
shape-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
dtype0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
[
lambda_2/percentile_2/q/xConst*
dtype0*
_output_shapes
: *
value	B :
z
lambda_2/percentile_2/qCastlambda_2/percentile_2/q/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
k
lambda_2/percentile_2/axisConst*
valueB"      *
dtype0*
_output_shapes
:
N
Flambda_2/percentile_2/assert_integer/statically_determined_was_integerNoOp
}
$lambda_2/percentile_2/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ѕ
lambda_2/percentile_2/transpose	Transposelambda_2/Placeholder$lambda_2/percentile_2/transpose/perm*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0*
T0
z
lambda_2/percentile_2/ShapeShapelambda_2/percentile_2/transpose*
T0*
out_type0*
_output_shapes
:
s
)lambda_2/percentile_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_2/percentile_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+lambda_2/percentile_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
л
#lambda_2/percentile_2/strided_sliceStridedSlicelambda_2/percentile_2/Shape)lambda_2/percentile_2/strided_slice/stack+lambda_2/percentile_2/strided_slice/stack_1+lambda_2/percentile_2/strided_slice/stack_2*
end_mask *
_output_shapes
:*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask 
x
%lambda_2/percentile_2/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
c
!lambda_2/percentile_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
—
lambda_2/percentile_2/concatConcatV2#lambda_2/percentile_2/strided_slice%lambda_2/percentile_2/concat/values_1!lambda_2/percentile_2/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
і
lambda_2/percentile_2/ReshapeReshapelambda_2/percentile_2/transposelambda_2/percentile_2/concat*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
h
lambda_2/percentile_2/truediv/yConst*
valueB 2      Y@*
dtype0*
_output_shapes
: 
Г
lambda_2/percentile_2/truedivRealDivlambda_2/percentile_2/qlambda_2/percentile_2/truediv/y*
T0*
_output_shapes
: 
d
lambda_2/percentile_2/sub/xConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
}
lambda_2/percentile_2/subSublambda_2/percentile_2/sub/xlambda_2/percentile_2/truediv*
T0*
_output_shapes
: 
z
lambda_2/percentile_2/Shape_1Shapelambda_2/percentile_2/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_2/percentile_2/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_2/percentile_2/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_2/percentile_2/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_2/percentile_2/strided_slice_1StridedSlicelambda_2/percentile_2/Shape_1+lambda_2/percentile_2/strided_slice_1/stack-lambda_2/percentile_2/strided_slice_1/stack_1-lambda_2/percentile_2/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
Н
lambda_2/percentile_2/ToDoubleCast%lambda_2/percentile_2/strided_slice_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
f
lambda_2/percentile_2/sub_1/yConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
В
lambda_2/percentile_2/sub_1Sublambda_2/percentile_2/ToDoublelambda_2/percentile_2/sub_1/y*
_output_shapes
: *
T0
y
lambda_2/percentile_2/mulMullambda_2/percentile_2/sub_1lambda_2/percentile_2/sub*
T0*
_output_shapes
: 
`
lambda_2/percentile_2/RoundRoundlambda_2/percentile_2/mul*
T0*
_output_shapes
: 
z
lambda_2/percentile_2/Shape_2Shapelambda_2/percentile_2/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_2/percentile_2/strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_2/percentile_2/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
w
-lambda_2/percentile_2/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_2/percentile_2/strided_slice_2StridedSlicelambda_2/percentile_2/Shape_2+lambda_2/percentile_2/strided_slice_2/stack-lambda_2/percentile_2/strided_slice_2/stack_1-lambda_2/percentile_2/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
В
lambda_2/percentile_2/ToInt32Castlambda_2/percentile_2/Round*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
_
lambda_2/percentile_2/sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Й
lambda_2/percentile_2/sub_2Sub%lambda_2/percentile_2/strided_slice_2lambda_2/percentile_2/sub_2/y*
T0*
_output_shapes
: 
У
+lambda_2/percentile_2/clip_by_value/MinimumMinimumlambda_2/percentile_2/ToInt32lambda_2/percentile_2/sub_2*
_output_shapes
: *
T0
g
%lambda_2/percentile_2/clip_by_value/yConst*
dtype0*
_output_shapes
: *
value	B : 
£
#lambda_2/percentile_2/clip_by_valueMaximum+lambda_2/percentile_2/clip_by_value/Minimum%lambda_2/percentile_2/clip_by_value/y*
T0*
_output_shapes
: 
z
lambda_2/percentile_2/Shape_3Shapelambda_2/percentile_2/Reshape*
_output_shapes
:*
T0*
out_type0
~
+lambda_2/percentile_2/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w
-lambda_2/percentile_2/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_2/percentile_2/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_2/percentile_2/strided_slice_3StridedSlicelambda_2/percentile_2/Shape_3+lambda_2/percentile_2/strided_slice_3/stack-lambda_2/percentile_2/strided_slice_3/stack_1-lambda_2/percentile_2/strided_slice_3/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
ў
lambda_2/percentile_2/TopKV2TopKV2lambda_2/percentile_2/Reshape%lambda_2/percentile_2/strided_slice_3*
sorted(*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€
]
lambda_2/percentile_2/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
Г
lambda_2/percentile_2/addAdd#lambda_2/percentile_2/clip_by_valuelambda_2/percentile_2/add/y*
T0*
_output_shapes
: 
o
-lambda_2/percentile_2/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
+lambda_2/percentile_2/strided_slice_4/stackPack-lambda_2/percentile_2/strided_slice_4/stack/0#lambda_2/percentile_2/clip_by_value*
T0*

axis *
N*
_output_shapes
:
q
/lambda_2/percentile_2/strided_slice_4/stack_1/0Const*
dtype0*
_output_shapes
: *
value	B : 
ї
-lambda_2/percentile_2/strided_slice_4/stack_1Pack/lambda_2/percentile_2/strided_slice_4/stack_1/0lambda_2/percentile_2/add*
T0*

axis *
N*
_output_shapes
:
~
-lambda_2/percentile_2/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
Б
%lambda_2/percentile_2/strided_slice_4StridedSlicelambda_2/percentile_2/TopKV2+lambda_2/percentile_2/strided_slice_4/stack-lambda_2/percentile_2/strided_slice_4/stack_1-lambda_2/percentile_2/strided_slice_4/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask*

begin_mask *
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€
f
$lambda_2/percentile_2/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
љ
 lambda_2/percentile_2/ExpandDims
ExpandDims%lambda_2/percentile_2/strided_slice_4$lambda_2/percentile_2/ExpandDims/dim*
T0*+
_output_shapes
:€€€€€€€€€*

Tdim0
h
&lambda_2/percentile_2/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
ј
"lambda_2/percentile_2/ExpandDims_1
ExpandDims lambda_2/percentile_2/ExpandDims&lambda_2/percentile_2/ExpandDims_1/dim*

Tdim0*
T0*/
_output_shapes
:€€€€€€€€€
^
lambda_2/percentile_3/q/xConst*
dtype0*
_output_shapes
: *
valueB
 *ЪЩ«B
z
lambda_2/percentile_3/qCastlambda_2/percentile_3/q/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
k
lambda_2/percentile_3/axisConst*
valueB"      *
dtype0*
_output_shapes
:
N
Flambda_2/percentile_3/assert_integer/statically_determined_was_integerNoOp
}
$lambda_2/percentile_3/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ѕ
lambda_2/percentile_3/transpose	Transposelambda_2/Placeholder$lambda_2/percentile_3/transpose/perm*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0*
T0
z
lambda_2/percentile_3/ShapeShapelambda_2/percentile_3/transpose*
T0*
out_type0*
_output_shapes
:
s
)lambda_2/percentile_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_2/percentile_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+lambda_2/percentile_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
л
#lambda_2/percentile_3/strided_sliceStridedSlicelambda_2/percentile_3/Shape)lambda_2/percentile_3/strided_slice/stack+lambda_2/percentile_3/strided_slice/stack_1+lambda_2/percentile_3/strided_slice/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask 
x
%lambda_2/percentile_3/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
c
!lambda_2/percentile_3/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
—
lambda_2/percentile_3/concatConcatV2#lambda_2/percentile_3/strided_slice%lambda_2/percentile_3/concat/values_1!lambda_2/percentile_3/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
і
lambda_2/percentile_3/ReshapeReshapelambda_2/percentile_3/transposelambda_2/percentile_3/concat*
T0*
Tshape0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
h
lambda_2/percentile_3/truediv/yConst*
dtype0*
_output_shapes
: *
valueB 2      Y@
Г
lambda_2/percentile_3/truedivRealDivlambda_2/percentile_3/qlambda_2/percentile_3/truediv/y*
T0*
_output_shapes
: 
d
lambda_2/percentile_3/sub/xConst*
dtype0*
_output_shapes
: *
valueB 2      р?
}
lambda_2/percentile_3/subSublambda_2/percentile_3/sub/xlambda_2/percentile_3/truediv*
T0*
_output_shapes
: 
z
lambda_2/percentile_3/Shape_1Shapelambda_2/percentile_3/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_2/percentile_3/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_2/percentile_3/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_2/percentile_3/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_2/percentile_3/strided_slice_1StridedSlicelambda_2/percentile_3/Shape_1+lambda_2/percentile_3/strided_slice_1/stack-lambda_2/percentile_3/strided_slice_1/stack_1-lambda_2/percentile_3/strided_slice_1/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
Н
lambda_2/percentile_3/ToDoubleCast%lambda_2/percentile_3/strided_slice_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
f
lambda_2/percentile_3/sub_1/yConst*
dtype0*
_output_shapes
: *
valueB 2      р?
В
lambda_2/percentile_3/sub_1Sublambda_2/percentile_3/ToDoublelambda_2/percentile_3/sub_1/y*
T0*
_output_shapes
: 
y
lambda_2/percentile_3/mulMullambda_2/percentile_3/sub_1lambda_2/percentile_3/sub*
T0*
_output_shapes
: 
`
lambda_2/percentile_3/RoundRoundlambda_2/percentile_3/mul*
T0*
_output_shapes
: 
z
lambda_2/percentile_3/Shape_2Shapelambda_2/percentile_3/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_2/percentile_3/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w
-lambda_2/percentile_3/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_2/percentile_3/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_2/percentile_3/strided_slice_2StridedSlicelambda_2/percentile_3/Shape_2+lambda_2/percentile_3/strided_slice_2/stack-lambda_2/percentile_3/strided_slice_2/stack_1-lambda_2/percentile_3/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
В
lambda_2/percentile_3/ToInt32Castlambda_2/percentile_3/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
_
lambda_2/percentile_3/sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Й
lambda_2/percentile_3/sub_2Sub%lambda_2/percentile_3/strided_slice_2lambda_2/percentile_3/sub_2/y*
T0*
_output_shapes
: 
У
+lambda_2/percentile_3/clip_by_value/MinimumMinimumlambda_2/percentile_3/ToInt32lambda_2/percentile_3/sub_2*
_output_shapes
: *
T0
g
%lambda_2/percentile_3/clip_by_value/yConst*
dtype0*
_output_shapes
: *
value	B : 
£
#lambda_2/percentile_3/clip_by_valueMaximum+lambda_2/percentile_3/clip_by_value/Minimum%lambda_2/percentile_3/clip_by_value/y*
_output_shapes
: *
T0
z
lambda_2/percentile_3/Shape_3Shapelambda_2/percentile_3/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_2/percentile_3/strided_slice_3/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_2/percentile_3/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_2/percentile_3/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_2/percentile_3/strided_slice_3StridedSlicelambda_2/percentile_3/Shape_3+lambda_2/percentile_3/strided_slice_3/stack-lambda_2/percentile_3/strided_slice_3/stack_1-lambda_2/percentile_3/strided_slice_3/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
ў
lambda_2/percentile_3/TopKV2TopKV2lambda_2/percentile_3/Reshape%lambda_2/percentile_3/strided_slice_3*
sorted(*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€
]
lambda_2/percentile_3/add/yConst*
dtype0*
_output_shapes
: *
value	B :
Г
lambda_2/percentile_3/addAdd#lambda_2/percentile_3/clip_by_valuelambda_2/percentile_3/add/y*
T0*
_output_shapes
: 
o
-lambda_2/percentile_3/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
+lambda_2/percentile_3/strided_slice_4/stackPack-lambda_2/percentile_3/strided_slice_4/stack/0#lambda_2/percentile_3/clip_by_value*
T0*

axis *
N*
_output_shapes
:
q
/lambda_2/percentile_3/strided_slice_4/stack_1/0Const*
value	B : *
dtype0*
_output_shapes
: 
ї
-lambda_2/percentile_3/strided_slice_4/stack_1Pack/lambda_2/percentile_3/strided_slice_4/stack_1/0lambda_2/percentile_3/add*
N*
_output_shapes
:*
T0*

axis 
~
-lambda_2/percentile_3/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Б
%lambda_2/percentile_3/strided_slice_4StridedSlicelambda_2/percentile_3/TopKV2+lambda_2/percentile_3/strided_slice_4/stack-lambda_2/percentile_3/strided_slice_4/stack_1-lambda_2/percentile_3/strided_slice_4/stack_2*

begin_mask *
ellipsis_mask*
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€*
T0*
Index0*
shrink_axis_mask
f
$lambda_2/percentile_3/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
љ
 lambda_2/percentile_3/ExpandDims
ExpandDims%lambda_2/percentile_3/strided_slice_4$lambda_2/percentile_3/ExpandDims/dim*
T0*+
_output_shapes
:€€€€€€€€€*

Tdim0
h
&lambda_2/percentile_3/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
ј
"lambda_2/percentile_3/ExpandDims_1
ExpandDims lambda_2/percentile_3/ExpandDims&lambda_2/percentile_3/ExpandDims_1/dim*

Tdim0*
T0*/
_output_shapes
:€€€€€€€€€
Ы
lambda_2/sub_2Sublambda_2/Placeholder"lambda_2/percentile_2/ExpandDims_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ч
lambda_2/sub_3Sub"lambda_2/percentile_3/ExpandDims_1"lambda_2/percentile_2/ExpandDims_1*
T0*/
_output_shapes
:€€€€€€€€€
U
lambda_2/add_1/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
q
lambda_2/add_1Addlambda_2/sub_3lambda_2/add_1/y*/
_output_shapes
:€€€€€€€€€*
T0
Й
lambda_2/truediv_1RealDivlambda_2/sub_2lambda_2/add_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
U
lambda_2/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
U
lambda_2/Const_3Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Э
 lambda_2/clip_by_value_1/MinimumMinimumlambda_2/truediv_1lambda_2/Const_3*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
£
lambda_2/clip_by_value_1Maximum lambda_2/clip_by_value_1/Minimumlambda_2/Const_2*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
W
net_input/tagConst*
valueB B	net_input*
dtype0*
_output_shapes
: 
Ф
	net_inputImageSummarynet_input/taglambda_2/clip_by_value*

max_images*
T0*
	bad_colorB:€  €*
_output_shapes
: 
_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
a
strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
≤
strided_slice_1StridedSlicePlaceholderstrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Y
lambda_3/percentile/q/xConst*
dtype0*
_output_shapes
: *
value	B :
v
lambda_3/percentile/qCastlambda_3/percentile/q/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
i
lambda_3/percentile/axisConst*
valueB"      *
dtype0*
_output_shapes
:
L
Dlambda_3/percentile/assert_integer/statically_determined_was_integerNoOp
{
"lambda_3/percentile/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Є
lambda_3/percentile/transpose	Transposestrided_slice_1"lambda_3/percentile/transpose/perm*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0
v
lambda_3/percentile/ShapeShapelambda_3/percentile/transpose*
T0*
out_type0*
_output_shapes
:
q
'lambda_3/percentile/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
s
)lambda_3/percentile/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)lambda_3/percentile/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
б
!lambda_3/percentile/strided_sliceStridedSlicelambda_3/percentile/Shape'lambda_3/percentile/strided_slice/stack)lambda_3/percentile/strided_slice/stack_1)lambda_3/percentile/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
v
#lambda_3/percentile/concat/values_1Const*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
a
lambda_3/percentile/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
…
lambda_3/percentile/concatConcatV2!lambda_3/percentile/strided_slice#lambda_3/percentile/concat/values_1lambda_3/percentile/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ѓ
lambda_3/percentile/ReshapeReshapelambda_3/percentile/transposelambda_3/percentile/concat*
T0*
Tshape0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
f
lambda_3/percentile/truediv/yConst*
valueB 2      Y@*
dtype0*
_output_shapes
: 
}
lambda_3/percentile/truedivRealDivlambda_3/percentile/qlambda_3/percentile/truediv/y*
T0*
_output_shapes
: 
b
lambda_3/percentile/sub/xConst*
dtype0*
_output_shapes
: *
valueB 2      р?
w
lambda_3/percentile/subSublambda_3/percentile/sub/xlambda_3/percentile/truediv*
T0*
_output_shapes
: 
v
lambda_3/percentile/Shape_1Shapelambda_3/percentile/Reshape*
T0*
out_type0*
_output_shapes
:
|
)lambda_3/percentile/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
u
+lambda_3/percentile/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_3/percentile/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
з
#lambda_3/percentile/strided_slice_1StridedSlicelambda_3/percentile/Shape_1)lambda_3/percentile/strided_slice_1/stack+lambda_3/percentile/strided_slice_1/stack_1+lambda_3/percentile/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Й
lambda_3/percentile/ToDoubleCast#lambda_3/percentile/strided_slice_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
d
lambda_3/percentile/sub_1/yConst*
dtype0*
_output_shapes
: *
valueB 2      р?
|
lambda_3/percentile/sub_1Sublambda_3/percentile/ToDoublelambda_3/percentile/sub_1/y*
_output_shapes
: *
T0
s
lambda_3/percentile/mulMullambda_3/percentile/sub_1lambda_3/percentile/sub*
T0*
_output_shapes
: 
\
lambda_3/percentile/RoundRoundlambda_3/percentile/mul*
_output_shapes
: *
T0
v
lambda_3/percentile/Shape_2Shapelambda_3/percentile/Reshape*
T0*
out_type0*
_output_shapes
:
|
)lambda_3/percentile/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
u
+lambda_3/percentile/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_3/percentile/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
з
#lambda_3/percentile/strided_slice_2StridedSlicelambda_3/percentile/Shape_2)lambda_3/percentile/strided_slice_2/stack+lambda_3/percentile/strided_slice_2/stack_1+lambda_3/percentile/strided_slice_2/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
~
lambda_3/percentile/ToInt32Castlambda_3/percentile/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
]
lambda_3/percentile/sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
Г
lambda_3/percentile/sub_2Sub#lambda_3/percentile/strided_slice_2lambda_3/percentile/sub_2/y*
T0*
_output_shapes
: 
Н
)lambda_3/percentile/clip_by_value/MinimumMinimumlambda_3/percentile/ToInt32lambda_3/percentile/sub_2*
T0*
_output_shapes
: 
e
#lambda_3/percentile/clip_by_value/yConst*
dtype0*
_output_shapes
: *
value	B : 
Э
!lambda_3/percentile/clip_by_valueMaximum)lambda_3/percentile/clip_by_value/Minimum#lambda_3/percentile/clip_by_value/y*
_output_shapes
: *
T0
v
lambda_3/percentile/Shape_3Shapelambda_3/percentile/Reshape*
T0*
out_type0*
_output_shapes
:
|
)lambda_3/percentile/strided_slice_3/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
u
+lambda_3/percentile/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_3/percentile/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
з
#lambda_3/percentile/strided_slice_3StridedSlicelambda_3/percentile/Shape_3)lambda_3/percentile/strided_slice_3/stack+lambda_3/percentile/strided_slice_3/stack_1+lambda_3/percentile/strided_slice_3/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
”
lambda_3/percentile/TopKV2TopKV2lambda_3/percentile/Reshape#lambda_3/percentile/strided_slice_3*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€*
sorted(
[
lambda_3/percentile/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
}
lambda_3/percentile/addAdd!lambda_3/percentile/clip_by_valuelambda_3/percentile/add/y*
T0*
_output_shapes
: 
m
+lambda_3/percentile/strided_slice_4/stack/0Const*
dtype0*
_output_shapes
: *
value	B : 
ї
)lambda_3/percentile/strided_slice_4/stackPack+lambda_3/percentile/strided_slice_4/stack/0!lambda_3/percentile/clip_by_value*
N*
_output_shapes
:*
T0*

axis 
o
-lambda_3/percentile/strided_slice_4/stack_1/0Const*
value	B : *
dtype0*
_output_shapes
: 
µ
+lambda_3/percentile/strided_slice_4/stack_1Pack-lambda_3/percentile/strided_slice_4/stack_1/0lambda_3/percentile/add*
N*
_output_shapes
:*
T0*

axis 
|
+lambda_3/percentile/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ч
#lambda_3/percentile/strided_slice_4StridedSlicelambda_3/percentile/TopKV2)lambda_3/percentile/strided_slice_4/stack+lambda_3/percentile/strided_slice_4/stack_1+lambda_3/percentile/strided_slice_4/stack_2*
end_mask *'
_output_shapes
:€€€€€€€€€*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask*
new_axis_mask 
d
"lambda_3/percentile/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ј
lambda_3/percentile/ExpandDims
ExpandDims#lambda_3/percentile/strided_slice_4"lambda_3/percentile/ExpandDims/dim*
T0*+
_output_shapes
:€€€€€€€€€*

Tdim0
f
$lambda_3/percentile/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ї
 lambda_3/percentile/ExpandDims_1
ExpandDimslambda_3/percentile/ExpandDims$lambda_3/percentile/ExpandDims_1/dim*
T0*/
_output_shapes
:€€€€€€€€€*

Tdim0
^
lambda_3/percentile_1/q/xConst*
valueB
 *ЪЩ«B*
dtype0*
_output_shapes
: 
z
lambda_3/percentile_1/qCastlambda_3/percentile_1/q/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
k
lambda_3/percentile_1/axisConst*
valueB"      *
dtype0*
_output_shapes
:
N
Flambda_3/percentile_1/assert_integer/statically_determined_was_integerNoOp
}
$lambda_3/percentile_1/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Љ
lambda_3/percentile_1/transpose	Transposestrided_slice_1$lambda_3/percentile_1/transpose/perm*
Tperm0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
z
lambda_3/percentile_1/ShapeShapelambda_3/percentile_1/transpose*
T0*
out_type0*
_output_shapes
:
s
)lambda_3/percentile_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_3/percentile_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+lambda_3/percentile_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
л
#lambda_3/percentile_1/strided_sliceStridedSlicelambda_3/percentile_1/Shape)lambda_3/percentile_1/strided_slice/stack+lambda_3/percentile_1/strided_slice/stack_1+lambda_3/percentile_1/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
:
x
%lambda_3/percentile_1/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
c
!lambda_3/percentile_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
—
lambda_3/percentile_1/concatConcatV2#lambda_3/percentile_1/strided_slice%lambda_3/percentile_1/concat/values_1!lambda_3/percentile_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
і
lambda_3/percentile_1/ReshapeReshapelambda_3/percentile_1/transposelambda_3/percentile_1/concat*
T0*
Tshape0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
h
lambda_3/percentile_1/truediv/yConst*
dtype0*
_output_shapes
: *
valueB 2      Y@
Г
lambda_3/percentile_1/truedivRealDivlambda_3/percentile_1/qlambda_3/percentile_1/truediv/y*
T0*
_output_shapes
: 
d
lambda_3/percentile_1/sub/xConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
}
lambda_3/percentile_1/subSublambda_3/percentile_1/sub/xlambda_3/percentile_1/truediv*
_output_shapes
: *
T0
z
lambda_3/percentile_1/Shape_1Shapelambda_3/percentile_1/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_3/percentile_1/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_3/percentile_1/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
w
-lambda_3/percentile_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_3/percentile_1/strided_slice_1StridedSlicelambda_3/percentile_1/Shape_1+lambda_3/percentile_1/strided_slice_1/stack-lambda_3/percentile_1/strided_slice_1/stack_1-lambda_3/percentile_1/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Н
lambda_3/percentile_1/ToDoubleCast%lambda_3/percentile_1/strided_slice_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
f
lambda_3/percentile_1/sub_1/yConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
В
lambda_3/percentile_1/sub_1Sublambda_3/percentile_1/ToDoublelambda_3/percentile_1/sub_1/y*
T0*
_output_shapes
: 
y
lambda_3/percentile_1/mulMullambda_3/percentile_1/sub_1lambda_3/percentile_1/sub*
_output_shapes
: *
T0
`
lambda_3/percentile_1/RoundRoundlambda_3/percentile_1/mul*
T0*
_output_shapes
: 
z
lambda_3/percentile_1/Shape_2Shapelambda_3/percentile_1/Reshape*
_output_shapes
:*
T0*
out_type0
~
+lambda_3/percentile_1/strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_3/percentile_1/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_3/percentile_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_3/percentile_1/strided_slice_2StridedSlicelambda_3/percentile_1/Shape_2+lambda_3/percentile_1/strided_slice_2/stack-lambda_3/percentile_1/strided_slice_2/stack_1-lambda_3/percentile_1/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
В
lambda_3/percentile_1/ToInt32Castlambda_3/percentile_1/Round*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
_
lambda_3/percentile_1/sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
Й
lambda_3/percentile_1/sub_2Sub%lambda_3/percentile_1/strided_slice_2lambda_3/percentile_1/sub_2/y*
_output_shapes
: *
T0
У
+lambda_3/percentile_1/clip_by_value/MinimumMinimumlambda_3/percentile_1/ToInt32lambda_3/percentile_1/sub_2*
_output_shapes
: *
T0
g
%lambda_3/percentile_1/clip_by_value/yConst*
value	B : *
dtype0*
_output_shapes
: 
£
#lambda_3/percentile_1/clip_by_valueMaximum+lambda_3/percentile_1/clip_by_value/Minimum%lambda_3/percentile_1/clip_by_value/y*
_output_shapes
: *
T0
z
lambda_3/percentile_1/Shape_3Shapelambda_3/percentile_1/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_3/percentile_1/strided_slice_3/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_3/percentile_1/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_3/percentile_1/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_3/percentile_1/strided_slice_3StridedSlicelambda_3/percentile_1/Shape_3+lambda_3/percentile_1/strided_slice_3/stack-lambda_3/percentile_1/strided_slice_3/stack_1-lambda_3/percentile_1/strided_slice_3/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
ў
lambda_3/percentile_1/TopKV2TopKV2lambda_3/percentile_1/Reshape%lambda_3/percentile_1/strided_slice_3*
sorted(*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€
]
lambda_3/percentile_1/add/yConst*
dtype0*
_output_shapes
: *
value	B :
Г
lambda_3/percentile_1/addAdd#lambda_3/percentile_1/clip_by_valuelambda_3/percentile_1/add/y*
T0*
_output_shapes
: 
o
-lambda_3/percentile_1/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
+lambda_3/percentile_1/strided_slice_4/stackPack-lambda_3/percentile_1/strided_slice_4/stack/0#lambda_3/percentile_1/clip_by_value*
T0*

axis *
N*
_output_shapes
:
q
/lambda_3/percentile_1/strided_slice_4/stack_1/0Const*
value	B : *
dtype0*
_output_shapes
: 
ї
-lambda_3/percentile_1/strided_slice_4/stack_1Pack/lambda_3/percentile_1/strided_slice_4/stack_1/0lambda_3/percentile_1/add*
N*
_output_shapes
:*
T0*

axis 
~
-lambda_3/percentile_1/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
Б
%lambda_3/percentile_1/strided_slice_4StridedSlicelambda_3/percentile_1/TopKV2+lambda_3/percentile_1/strided_slice_4/stack-lambda_3/percentile_1/strided_slice_4/stack_1-lambda_3/percentile_1/strided_slice_4/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask*

begin_mask *
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€
f
$lambda_3/percentile_1/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
љ
 lambda_3/percentile_1/ExpandDims
ExpandDims%lambda_3/percentile_1/strided_slice_4$lambda_3/percentile_1/ExpandDims/dim*

Tdim0*
T0*+
_output_shapes
:€€€€€€€€€
h
&lambda_3/percentile_1/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
ј
"lambda_3/percentile_1/ExpandDims_1
ExpandDims lambda_3/percentile_1/ExpandDims&lambda_3/percentile_1/ExpandDims_1/dim*/
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
Т
lambda_3/subSubstrided_slice_1 lambda_3/percentile/ExpandDims_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Х
lambda_3/sub_1Sub"lambda_3/percentile_1/ExpandDims_1 lambda_3/percentile/ExpandDims_1*
T0*/
_output_shapes
:€€€€€€€€€
S
lambda_3/add/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
m
lambda_3/addAddlambda_3/sub_1lambda_3/add/y*
T0*/
_output_shapes
:€€€€€€€€€
Г
lambda_3/truedivRealDivlambda_3/sublambda_3/add*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
S
lambda_3/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
U
lambda_3/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Щ
lambda_3/clip_by_value/MinimumMinimumlambda_3/truedivlambda_3/Const_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Э
lambda_3/clip_by_valueMaximumlambda_3/clip_by_value/Minimumlambda_3/Const*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ђ
lambda_3/PlaceholderPlaceholder*
dtype0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*6
shape-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
[
lambda_3/percentile_2/q/xConst*
value	B :*
dtype0*
_output_shapes
: 
z
lambda_3/percentile_2/qCastlambda_3/percentile_2/q/x*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
k
lambda_3/percentile_2/axisConst*
valueB"      *
dtype0*
_output_shapes
:
N
Flambda_3/percentile_2/assert_integer/statically_determined_was_integerNoOp
}
$lambda_3/percentile_2/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ѕ
lambda_3/percentile_2/transpose	Transposelambda_3/Placeholder$lambda_3/percentile_2/transpose/perm*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0*
T0
z
lambda_3/percentile_2/ShapeShapelambda_3/percentile_2/transpose*
T0*
out_type0*
_output_shapes
:
s
)lambda_3/percentile_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_3/percentile_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+lambda_3/percentile_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
л
#lambda_3/percentile_2/strided_sliceStridedSlicelambda_3/percentile_2/Shape)lambda_3/percentile_2/strided_slice/stack+lambda_3/percentile_2/strided_slice/stack_1+lambda_3/percentile_2/strided_slice/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
x
%lambda_3/percentile_2/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
c
!lambda_3/percentile_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
—
lambda_3/percentile_2/concatConcatV2#lambda_3/percentile_2/strided_slice%lambda_3/percentile_2/concat/values_1!lambda_3/percentile_2/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
і
lambda_3/percentile_2/ReshapeReshapelambda_3/percentile_2/transposelambda_3/percentile_2/concat*
T0*
Tshape0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
h
lambda_3/percentile_2/truediv/yConst*
valueB 2      Y@*
dtype0*
_output_shapes
: 
Г
lambda_3/percentile_2/truedivRealDivlambda_3/percentile_2/qlambda_3/percentile_2/truediv/y*
T0*
_output_shapes
: 
d
lambda_3/percentile_2/sub/xConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
}
lambda_3/percentile_2/subSublambda_3/percentile_2/sub/xlambda_3/percentile_2/truediv*
T0*
_output_shapes
: 
z
lambda_3/percentile_2/Shape_1Shapelambda_3/percentile_2/Reshape*
_output_shapes
:*
T0*
out_type0
~
+lambda_3/percentile_2/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_3/percentile_2/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_3/percentile_2/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_3/percentile_2/strided_slice_1StridedSlicelambda_3/percentile_2/Shape_1+lambda_3/percentile_2/strided_slice_1/stack-lambda_3/percentile_2/strided_slice_1/stack_1-lambda_3/percentile_2/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
Н
lambda_3/percentile_2/ToDoubleCast%lambda_3/percentile_2/strided_slice_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
f
lambda_3/percentile_2/sub_1/yConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
В
lambda_3/percentile_2/sub_1Sublambda_3/percentile_2/ToDoublelambda_3/percentile_2/sub_1/y*
T0*
_output_shapes
: 
y
lambda_3/percentile_2/mulMullambda_3/percentile_2/sub_1lambda_3/percentile_2/sub*
T0*
_output_shapes
: 
`
lambda_3/percentile_2/RoundRoundlambda_3/percentile_2/mul*
_output_shapes
: *
T0
z
lambda_3/percentile_2/Shape_2Shapelambda_3/percentile_2/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_3/percentile_2/strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_3/percentile_2/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
w
-lambda_3/percentile_2/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_3/percentile_2/strided_slice_2StridedSlicelambda_3/percentile_2/Shape_2+lambda_3/percentile_2/strided_slice_2/stack-lambda_3/percentile_2/strided_slice_2/stack_1-lambda_3/percentile_2/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
В
lambda_3/percentile_2/ToInt32Castlambda_3/percentile_2/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
_
lambda_3/percentile_2/sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Й
lambda_3/percentile_2/sub_2Sub%lambda_3/percentile_2/strided_slice_2lambda_3/percentile_2/sub_2/y*
_output_shapes
: *
T0
У
+lambda_3/percentile_2/clip_by_value/MinimumMinimumlambda_3/percentile_2/ToInt32lambda_3/percentile_2/sub_2*
_output_shapes
: *
T0
g
%lambda_3/percentile_2/clip_by_value/yConst*
value	B : *
dtype0*
_output_shapes
: 
£
#lambda_3/percentile_2/clip_by_valueMaximum+lambda_3/percentile_2/clip_by_value/Minimum%lambda_3/percentile_2/clip_by_value/y*
T0*
_output_shapes
: 
z
lambda_3/percentile_2/Shape_3Shapelambda_3/percentile_2/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_3/percentile_2/strided_slice_3/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_3/percentile_2/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_3/percentile_2/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_3/percentile_2/strided_slice_3StridedSlicelambda_3/percentile_2/Shape_3+lambda_3/percentile_2/strided_slice_3/stack-lambda_3/percentile_2/strided_slice_3/stack_1-lambda_3/percentile_2/strided_slice_3/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
ў
lambda_3/percentile_2/TopKV2TopKV2lambda_3/percentile_2/Reshape%lambda_3/percentile_2/strided_slice_3*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€*
sorted(
]
lambda_3/percentile_2/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
Г
lambda_3/percentile_2/addAdd#lambda_3/percentile_2/clip_by_valuelambda_3/percentile_2/add/y*
T0*
_output_shapes
: 
o
-lambda_3/percentile_2/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
+lambda_3/percentile_2/strided_slice_4/stackPack-lambda_3/percentile_2/strided_slice_4/stack/0#lambda_3/percentile_2/clip_by_value*
T0*

axis *
N*
_output_shapes
:
q
/lambda_3/percentile_2/strided_slice_4/stack_1/0Const*
value	B : *
dtype0*
_output_shapes
: 
ї
-lambda_3/percentile_2/strided_slice_4/stack_1Pack/lambda_3/percentile_2/strided_slice_4/stack_1/0lambda_3/percentile_2/add*
T0*

axis *
N*
_output_shapes
:
~
-lambda_3/percentile_2/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Б
%lambda_3/percentile_2/strided_slice_4StridedSlicelambda_3/percentile_2/TopKV2+lambda_3/percentile_2/strided_slice_4/stack-lambda_3/percentile_2/strided_slice_4/stack_1-lambda_3/percentile_2/strided_slice_4/stack_2*
end_mask *'
_output_shapes
:€€€€€€€€€*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask*

begin_mask *
new_axis_mask 
f
$lambda_3/percentile_2/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
љ
 lambda_3/percentile_2/ExpandDims
ExpandDims%lambda_3/percentile_2/strided_slice_4$lambda_3/percentile_2/ExpandDims/dim*

Tdim0*
T0*+
_output_shapes
:€€€€€€€€€
h
&lambda_3/percentile_2/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
ј
"lambda_3/percentile_2/ExpandDims_1
ExpandDims lambda_3/percentile_2/ExpandDims&lambda_3/percentile_2/ExpandDims_1/dim*

Tdim0*
T0*/
_output_shapes
:€€€€€€€€€
^
lambda_3/percentile_3/q/xConst*
valueB
 *ЪЩ«B*
dtype0*
_output_shapes
: 
z
lambda_3/percentile_3/qCastlambda_3/percentile_3/q/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
k
lambda_3/percentile_3/axisConst*
dtype0*
_output_shapes
:*
valueB"      
N
Flambda_3/percentile_3/assert_integer/statically_determined_was_integerNoOp
}
$lambda_3/percentile_3/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ѕ
lambda_3/percentile_3/transpose	Transposelambda_3/Placeholder$lambda_3/percentile_3/transpose/perm*
Tperm0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
z
lambda_3/percentile_3/ShapeShapelambda_3/percentile_3/transpose*
_output_shapes
:*
T0*
out_type0
s
)lambda_3/percentile_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_3/percentile_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+lambda_3/percentile_3/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
л
#lambda_3/percentile_3/strided_sliceStridedSlicelambda_3/percentile_3/Shape)lambda_3/percentile_3/strided_slice/stack+lambda_3/percentile_3/strided_slice/stack_1+lambda_3/percentile_3/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
x
%lambda_3/percentile_3/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
c
!lambda_3/percentile_3/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
—
lambda_3/percentile_3/concatConcatV2#lambda_3/percentile_3/strided_slice%lambda_3/percentile_3/concat/values_1!lambda_3/percentile_3/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
і
lambda_3/percentile_3/ReshapeReshapelambda_3/percentile_3/transposelambda_3/percentile_3/concat*
T0*
Tshape0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
h
lambda_3/percentile_3/truediv/yConst*
dtype0*
_output_shapes
: *
valueB 2      Y@
Г
lambda_3/percentile_3/truedivRealDivlambda_3/percentile_3/qlambda_3/percentile_3/truediv/y*
_output_shapes
: *
T0
d
lambda_3/percentile_3/sub/xConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
}
lambda_3/percentile_3/subSublambda_3/percentile_3/sub/xlambda_3/percentile_3/truediv*
T0*
_output_shapes
: 
z
lambda_3/percentile_3/Shape_1Shapelambda_3/percentile_3/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_3/percentile_3/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_3/percentile_3/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_3/percentile_3/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_3/percentile_3/strided_slice_1StridedSlicelambda_3/percentile_3/Shape_1+lambda_3/percentile_3/strided_slice_1/stack-lambda_3/percentile_3/strided_slice_1/stack_1-lambda_3/percentile_3/strided_slice_1/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
Н
lambda_3/percentile_3/ToDoubleCast%lambda_3/percentile_3/strided_slice_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
f
lambda_3/percentile_3/sub_1/yConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
В
lambda_3/percentile_3/sub_1Sublambda_3/percentile_3/ToDoublelambda_3/percentile_3/sub_1/y*
T0*
_output_shapes
: 
y
lambda_3/percentile_3/mulMullambda_3/percentile_3/sub_1lambda_3/percentile_3/sub*
T0*
_output_shapes
: 
`
lambda_3/percentile_3/RoundRoundlambda_3/percentile_3/mul*
_output_shapes
: *
T0
z
lambda_3/percentile_3/Shape_2Shapelambda_3/percentile_3/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_3/percentile_3/strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_3/percentile_3/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
w
-lambda_3/percentile_3/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_3/percentile_3/strided_slice_2StridedSlicelambda_3/percentile_3/Shape_2+lambda_3/percentile_3/strided_slice_2/stack-lambda_3/percentile_3/strided_slice_2/stack_1-lambda_3/percentile_3/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
В
lambda_3/percentile_3/ToInt32Castlambda_3/percentile_3/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
_
lambda_3/percentile_3/sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Й
lambda_3/percentile_3/sub_2Sub%lambda_3/percentile_3/strided_slice_2lambda_3/percentile_3/sub_2/y*
_output_shapes
: *
T0
У
+lambda_3/percentile_3/clip_by_value/MinimumMinimumlambda_3/percentile_3/ToInt32lambda_3/percentile_3/sub_2*
T0*
_output_shapes
: 
g
%lambda_3/percentile_3/clip_by_value/yConst*
value	B : *
dtype0*
_output_shapes
: 
£
#lambda_3/percentile_3/clip_by_valueMaximum+lambda_3/percentile_3/clip_by_value/Minimum%lambda_3/percentile_3/clip_by_value/y*
T0*
_output_shapes
: 
z
lambda_3/percentile_3/Shape_3Shapelambda_3/percentile_3/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_3/percentile_3/strided_slice_3/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_3/percentile_3/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_3/percentile_3/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_3/percentile_3/strided_slice_3StridedSlicelambda_3/percentile_3/Shape_3+lambda_3/percentile_3/strided_slice_3/stack-lambda_3/percentile_3/strided_slice_3/stack_1-lambda_3/percentile_3/strided_slice_3/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
ў
lambda_3/percentile_3/TopKV2TopKV2lambda_3/percentile_3/Reshape%lambda_3/percentile_3/strided_slice_3*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€*
sorted(
]
lambda_3/percentile_3/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
Г
lambda_3/percentile_3/addAdd#lambda_3/percentile_3/clip_by_valuelambda_3/percentile_3/add/y*
_output_shapes
: *
T0
o
-lambda_3/percentile_3/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
+lambda_3/percentile_3/strided_slice_4/stackPack-lambda_3/percentile_3/strided_slice_4/stack/0#lambda_3/percentile_3/clip_by_value*
N*
_output_shapes
:*
T0*

axis 
q
/lambda_3/percentile_3/strided_slice_4/stack_1/0Const*
value	B : *
dtype0*
_output_shapes
: 
ї
-lambda_3/percentile_3/strided_slice_4/stack_1Pack/lambda_3/percentile_3/strided_slice_4/stack_1/0lambda_3/percentile_3/add*
T0*

axis *
N*
_output_shapes
:
~
-lambda_3/percentile_3/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
Б
%lambda_3/percentile_3/strided_slice_4StridedSlicelambda_3/percentile_3/TopKV2+lambda_3/percentile_3/strided_slice_4/stack-lambda_3/percentile_3/strided_slice_4/stack_1-lambda_3/percentile_3/strided_slice_4/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask*
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€*
Index0*
T0
f
$lambda_3/percentile_3/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
љ
 lambda_3/percentile_3/ExpandDims
ExpandDims%lambda_3/percentile_3/strided_slice_4$lambda_3/percentile_3/ExpandDims/dim*+
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
h
&lambda_3/percentile_3/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B :
ј
"lambda_3/percentile_3/ExpandDims_1
ExpandDims lambda_3/percentile_3/ExpandDims&lambda_3/percentile_3/ExpandDims_1/dim*

Tdim0*
T0*/
_output_shapes
:€€€€€€€€€
Ы
lambda_3/sub_2Sublambda_3/Placeholder"lambda_3/percentile_2/ExpandDims_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ч
lambda_3/sub_3Sub"lambda_3/percentile_3/ExpandDims_1"lambda_3/percentile_2/ExpandDims_1*
T0*/
_output_shapes
:€€€€€€€€€
U
lambda_3/add_1/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
q
lambda_3/add_1Addlambda_3/sub_3lambda_3/add_1/y*/
_output_shapes
:€€€€€€€€€*
T0
Й
lambda_3/truediv_1RealDivlambda_3/sub_2lambda_3/add_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
U
lambda_3/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
U
lambda_3/Const_3Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Э
 lambda_3/clip_by_value_1/MinimumMinimumlambda_3/truediv_1lambda_3/Const_3*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
£
lambda_3/clip_by_value_1Maximum lambda_3/clip_by_value_1/Minimumlambda_3/Const_2*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Y
net_target/tagConst*
valueB B
net_target*
dtype0*
_output_shapes
: 
Ц

net_targetImageSummarynet_target/taglambda_3/clip_by_value*

max_images*
T0*
	bad_colorB:€  €*
_output_shapes
: 
f
strided_slice_2/stackConst*
valueB"        *
dtype0*
_output_shapes
:
h
strided_slice_2/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
h
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
ї
strided_slice_2StridedSliceconcatenate_3/concatstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
shrink_axis_mask *
ellipsis_mask*

begin_mask*
new_axis_mask *
end_mask *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Index0*
T0
_
strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ґ
strided_slice_3StridedSlicestrided_slice_2strided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Y
lambda_4/percentile/q/xConst*
value	B :*
dtype0*
_output_shapes
: 
v
lambda_4/percentile/qCastlambda_4/percentile/q/x*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
i
lambda_4/percentile/axisConst*
dtype0*
_output_shapes
:*
valueB"      
L
Dlambda_4/percentile/assert_integer/statically_determined_was_integerNoOp
{
"lambda_4/percentile/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Є
lambda_4/percentile/transpose	Transposestrided_slice_3"lambda_4/percentile/transpose/perm*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0*
T0
v
lambda_4/percentile/ShapeShapelambda_4/percentile/transpose*
T0*
out_type0*
_output_shapes
:
q
'lambda_4/percentile/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
s
)lambda_4/percentile/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)lambda_4/percentile/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
б
!lambda_4/percentile/strided_sliceStridedSlicelambda_4/percentile/Shape'lambda_4/percentile/strided_slice/stack)lambda_4/percentile/strided_slice/stack_1)lambda_4/percentile/strided_slice/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
v
#lambda_4/percentile/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
a
lambda_4/percentile/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
…
lambda_4/percentile/concatConcatV2!lambda_4/percentile/strided_slice#lambda_4/percentile/concat/values_1lambda_4/percentile/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ѓ
lambda_4/percentile/ReshapeReshapelambda_4/percentile/transposelambda_4/percentile/concat*
T0*
Tshape0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
f
lambda_4/percentile/truediv/yConst*
valueB 2      Y@*
dtype0*
_output_shapes
: 
}
lambda_4/percentile/truedivRealDivlambda_4/percentile/qlambda_4/percentile/truediv/y*
T0*
_output_shapes
: 
b
lambda_4/percentile/sub/xConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
w
lambda_4/percentile/subSublambda_4/percentile/sub/xlambda_4/percentile/truediv*
T0*
_output_shapes
: 
v
lambda_4/percentile/Shape_1Shapelambda_4/percentile/Reshape*
T0*
out_type0*
_output_shapes
:
|
)lambda_4/percentile/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
u
+lambda_4/percentile/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_4/percentile/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
з
#lambda_4/percentile/strided_slice_1StridedSlicelambda_4/percentile/Shape_1)lambda_4/percentile/strided_slice_1/stack+lambda_4/percentile/strided_slice_1/stack_1+lambda_4/percentile/strided_slice_1/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
Й
lambda_4/percentile/ToDoubleCast#lambda_4/percentile/strided_slice_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
d
lambda_4/percentile/sub_1/yConst*
dtype0*
_output_shapes
: *
valueB 2      р?
|
lambda_4/percentile/sub_1Sublambda_4/percentile/ToDoublelambda_4/percentile/sub_1/y*
T0*
_output_shapes
: 
s
lambda_4/percentile/mulMullambda_4/percentile/sub_1lambda_4/percentile/sub*
_output_shapes
: *
T0
\
lambda_4/percentile/RoundRoundlambda_4/percentile/mul*
T0*
_output_shapes
: 
v
lambda_4/percentile/Shape_2Shapelambda_4/percentile/Reshape*
T0*
out_type0*
_output_shapes
:
|
)lambda_4/percentile/strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
u
+lambda_4/percentile/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_4/percentile/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
з
#lambda_4/percentile/strided_slice_2StridedSlicelambda_4/percentile/Shape_2)lambda_4/percentile/strided_slice_2/stack+lambda_4/percentile/strided_slice_2/stack_1+lambda_4/percentile/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
~
lambda_4/percentile/ToInt32Castlambda_4/percentile/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
]
lambda_4/percentile/sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Г
lambda_4/percentile/sub_2Sub#lambda_4/percentile/strided_slice_2lambda_4/percentile/sub_2/y*
T0*
_output_shapes
: 
Н
)lambda_4/percentile/clip_by_value/MinimumMinimumlambda_4/percentile/ToInt32lambda_4/percentile/sub_2*
T0*
_output_shapes
: 
e
#lambda_4/percentile/clip_by_value/yConst*
value	B : *
dtype0*
_output_shapes
: 
Э
!lambda_4/percentile/clip_by_valueMaximum)lambda_4/percentile/clip_by_value/Minimum#lambda_4/percentile/clip_by_value/y*
_output_shapes
: *
T0
v
lambda_4/percentile/Shape_3Shapelambda_4/percentile/Reshape*
T0*
out_type0*
_output_shapes
:
|
)lambda_4/percentile/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
u
+lambda_4/percentile/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_4/percentile/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
з
#lambda_4/percentile/strided_slice_3StridedSlicelambda_4/percentile/Shape_3)lambda_4/percentile/strided_slice_3/stack+lambda_4/percentile/strided_slice_3/stack_1+lambda_4/percentile/strided_slice_3/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
”
lambda_4/percentile/TopKV2TopKV2lambda_4/percentile/Reshape#lambda_4/percentile/strided_slice_3*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€*
sorted(
[
lambda_4/percentile/add/yConst*
dtype0*
_output_shapes
: *
value	B :
}
lambda_4/percentile/addAdd!lambda_4/percentile/clip_by_valuelambda_4/percentile/add/y*
T0*
_output_shapes
: 
m
+lambda_4/percentile/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
ї
)lambda_4/percentile/strided_slice_4/stackPack+lambda_4/percentile/strided_slice_4/stack/0!lambda_4/percentile/clip_by_value*
T0*

axis *
N*
_output_shapes
:
o
-lambda_4/percentile/strided_slice_4/stack_1/0Const*
dtype0*
_output_shapes
: *
value	B : 
µ
+lambda_4/percentile/strided_slice_4/stack_1Pack-lambda_4/percentile/strided_slice_4/stack_1/0lambda_4/percentile/add*
T0*

axis *
N*
_output_shapes
:
|
+lambda_4/percentile/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ч
#lambda_4/percentile/strided_slice_4StridedSlicelambda_4/percentile/TopKV2)lambda_4/percentile/strided_slice_4/stack+lambda_4/percentile/strided_slice_4/stack_1+lambda_4/percentile/strided_slice_4/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask*
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€*
T0*
Index0
d
"lambda_4/percentile/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ј
lambda_4/percentile/ExpandDims
ExpandDims#lambda_4/percentile/strided_slice_4"lambda_4/percentile/ExpandDims/dim*+
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
f
$lambda_4/percentile/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
Ї
 lambda_4/percentile/ExpandDims_1
ExpandDimslambda_4/percentile/ExpandDims$lambda_4/percentile/ExpandDims_1/dim*

Tdim0*
T0*/
_output_shapes
:€€€€€€€€€
^
lambda_4/percentile_1/q/xConst*
valueB
 *ЪЩ«B*
dtype0*
_output_shapes
: 
z
lambda_4/percentile_1/qCastlambda_4/percentile_1/q/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
k
lambda_4/percentile_1/axisConst*
valueB"      *
dtype0*
_output_shapes
:
N
Flambda_4/percentile_1/assert_integer/statically_determined_was_integerNoOp
}
$lambda_4/percentile_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Љ
lambda_4/percentile_1/transpose	Transposestrided_slice_3$lambda_4/percentile_1/transpose/perm*
Tperm0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
z
lambda_4/percentile_1/ShapeShapelambda_4/percentile_1/transpose*
T0*
out_type0*
_output_shapes
:
s
)lambda_4/percentile_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_4/percentile_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+lambda_4/percentile_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
л
#lambda_4/percentile_1/strided_sliceStridedSlicelambda_4/percentile_1/Shape)lambda_4/percentile_1/strided_slice/stack+lambda_4/percentile_1/strided_slice/stack_1+lambda_4/percentile_1/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
x
%lambda_4/percentile_1/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
c
!lambda_4/percentile_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
—
lambda_4/percentile_1/concatConcatV2#lambda_4/percentile_1/strided_slice%lambda_4/percentile_1/concat/values_1!lambda_4/percentile_1/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
і
lambda_4/percentile_1/ReshapeReshapelambda_4/percentile_1/transposelambda_4/percentile_1/concat*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
h
lambda_4/percentile_1/truediv/yConst*
valueB 2      Y@*
dtype0*
_output_shapes
: 
Г
lambda_4/percentile_1/truedivRealDivlambda_4/percentile_1/qlambda_4/percentile_1/truediv/y*
T0*
_output_shapes
: 
d
lambda_4/percentile_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB 2      р?
}
lambda_4/percentile_1/subSublambda_4/percentile_1/sub/xlambda_4/percentile_1/truediv*
T0*
_output_shapes
: 
z
lambda_4/percentile_1/Shape_1Shapelambda_4/percentile_1/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_4/percentile_1/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_4/percentile_1/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_4/percentile_1/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_4/percentile_1/strided_slice_1StridedSlicelambda_4/percentile_1/Shape_1+lambda_4/percentile_1/strided_slice_1/stack-lambda_4/percentile_1/strided_slice_1/stack_1-lambda_4/percentile_1/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Н
lambda_4/percentile_1/ToDoubleCast%lambda_4/percentile_1/strided_slice_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
f
lambda_4/percentile_1/sub_1/yConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
В
lambda_4/percentile_1/sub_1Sublambda_4/percentile_1/ToDoublelambda_4/percentile_1/sub_1/y*
T0*
_output_shapes
: 
y
lambda_4/percentile_1/mulMullambda_4/percentile_1/sub_1lambda_4/percentile_1/sub*
T0*
_output_shapes
: 
`
lambda_4/percentile_1/RoundRoundlambda_4/percentile_1/mul*
T0*
_output_shapes
: 
z
lambda_4/percentile_1/Shape_2Shapelambda_4/percentile_1/Reshape*
_output_shapes
:*
T0*
out_type0
~
+lambda_4/percentile_1/strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_4/percentile_1/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_4/percentile_1/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_4/percentile_1/strided_slice_2StridedSlicelambda_4/percentile_1/Shape_2+lambda_4/percentile_1/strided_slice_2/stack-lambda_4/percentile_1/strided_slice_2/stack_1-lambda_4/percentile_1/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
В
lambda_4/percentile_1/ToInt32Castlambda_4/percentile_1/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
_
lambda_4/percentile_1/sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
Й
lambda_4/percentile_1/sub_2Sub%lambda_4/percentile_1/strided_slice_2lambda_4/percentile_1/sub_2/y*
T0*
_output_shapes
: 
У
+lambda_4/percentile_1/clip_by_value/MinimumMinimumlambda_4/percentile_1/ToInt32lambda_4/percentile_1/sub_2*
T0*
_output_shapes
: 
g
%lambda_4/percentile_1/clip_by_value/yConst*
value	B : *
dtype0*
_output_shapes
: 
£
#lambda_4/percentile_1/clip_by_valueMaximum+lambda_4/percentile_1/clip_by_value/Minimum%lambda_4/percentile_1/clip_by_value/y*
T0*
_output_shapes
: 
z
lambda_4/percentile_1/Shape_3Shapelambda_4/percentile_1/Reshape*
_output_shapes
:*
T0*
out_type0
~
+lambda_4/percentile_1/strided_slice_3/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_4/percentile_1/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_4/percentile_1/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_4/percentile_1/strided_slice_3StridedSlicelambda_4/percentile_1/Shape_3+lambda_4/percentile_1/strided_slice_3/stack-lambda_4/percentile_1/strided_slice_3/stack_1-lambda_4/percentile_1/strided_slice_3/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
ў
lambda_4/percentile_1/TopKV2TopKV2lambda_4/percentile_1/Reshape%lambda_4/percentile_1/strided_slice_3*
sorted(*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€
]
lambda_4/percentile_1/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
Г
lambda_4/percentile_1/addAdd#lambda_4/percentile_1/clip_by_valuelambda_4/percentile_1/add/y*
T0*
_output_shapes
: 
o
-lambda_4/percentile_1/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
+lambda_4/percentile_1/strided_slice_4/stackPack-lambda_4/percentile_1/strided_slice_4/stack/0#lambda_4/percentile_1/clip_by_value*
T0*

axis *
N*
_output_shapes
:
q
/lambda_4/percentile_1/strided_slice_4/stack_1/0Const*
value	B : *
dtype0*
_output_shapes
: 
ї
-lambda_4/percentile_1/strided_slice_4/stack_1Pack/lambda_4/percentile_1/strided_slice_4/stack_1/0lambda_4/percentile_1/add*
T0*

axis *
N*
_output_shapes
:
~
-lambda_4/percentile_1/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Б
%lambda_4/percentile_1/strided_slice_4StridedSlicelambda_4/percentile_1/TopKV2+lambda_4/percentile_1/strided_slice_4/stack-lambda_4/percentile_1/strided_slice_4/stack_1-lambda_4/percentile_1/strided_slice_4/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask*
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€
f
$lambda_4/percentile_1/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
љ
 lambda_4/percentile_1/ExpandDims
ExpandDims%lambda_4/percentile_1/strided_slice_4$lambda_4/percentile_1/ExpandDims/dim*+
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
h
&lambda_4/percentile_1/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
ј
"lambda_4/percentile_1/ExpandDims_1
ExpandDims lambda_4/percentile_1/ExpandDims&lambda_4/percentile_1/ExpandDims_1/dim*

Tdim0*
T0*/
_output_shapes
:€€€€€€€€€
Т
lambda_4/subSubstrided_slice_3 lambda_4/percentile/ExpandDims_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Х
lambda_4/sub_1Sub"lambda_4/percentile_1/ExpandDims_1 lambda_4/percentile/ExpandDims_1*
T0*/
_output_shapes
:€€€€€€€€€
S
lambda_4/add/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
m
lambda_4/addAddlambda_4/sub_1lambda_4/add/y*
T0*/
_output_shapes
:€€€€€€€€€
Г
lambda_4/truedivRealDivlambda_4/sublambda_4/add*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
S
lambda_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
U
lambda_4/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Щ
lambda_4/clip_by_value/MinimumMinimumlambda_4/truedivlambda_4/Const_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Э
lambda_4/clip_by_valueMaximumlambda_4/clip_by_value/Minimumlambda_4/Const*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0
Ђ
lambda_4/PlaceholderPlaceholder*6
shape-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
dtype0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
[
lambda_4/percentile_2/q/xConst*
value	B :*
dtype0*
_output_shapes
: 
z
lambda_4/percentile_2/qCastlambda_4/percentile_2/q/x*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
k
lambda_4/percentile_2/axisConst*
dtype0*
_output_shapes
:*
valueB"      
N
Flambda_4/percentile_2/assert_integer/statically_determined_was_integerNoOp
}
$lambda_4/percentile_2/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ѕ
lambda_4/percentile_2/transpose	Transposelambda_4/Placeholder$lambda_4/percentile_2/transpose/perm*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0
z
lambda_4/percentile_2/ShapeShapelambda_4/percentile_2/transpose*
_output_shapes
:*
T0*
out_type0
s
)lambda_4/percentile_2/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
u
+lambda_4/percentile_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+lambda_4/percentile_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
л
#lambda_4/percentile_2/strided_sliceStridedSlicelambda_4/percentile_2/Shape)lambda_4/percentile_2/strided_slice/stack+lambda_4/percentile_2/strided_slice/stack_1+lambda_4/percentile_2/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
x
%lambda_4/percentile_2/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
c
!lambda_4/percentile_2/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
—
lambda_4/percentile_2/concatConcatV2#lambda_4/percentile_2/strided_slice%lambda_4/percentile_2/concat/values_1!lambda_4/percentile_2/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
і
lambda_4/percentile_2/ReshapeReshapelambda_4/percentile_2/transposelambda_4/percentile_2/concat*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
h
lambda_4/percentile_2/truediv/yConst*
dtype0*
_output_shapes
: *
valueB 2      Y@
Г
lambda_4/percentile_2/truedivRealDivlambda_4/percentile_2/qlambda_4/percentile_2/truediv/y*
_output_shapes
: *
T0
d
lambda_4/percentile_2/sub/xConst*
dtype0*
_output_shapes
: *
valueB 2      р?
}
lambda_4/percentile_2/subSublambda_4/percentile_2/sub/xlambda_4/percentile_2/truediv*
T0*
_output_shapes
: 
z
lambda_4/percentile_2/Shape_1Shapelambda_4/percentile_2/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_4/percentile_2/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w
-lambda_4/percentile_2/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_4/percentile_2/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_4/percentile_2/strided_slice_1StridedSlicelambda_4/percentile_2/Shape_1+lambda_4/percentile_2/strided_slice_1/stack-lambda_4/percentile_2/strided_slice_1/stack_1-lambda_4/percentile_2/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Н
lambda_4/percentile_2/ToDoubleCast%lambda_4/percentile_2/strided_slice_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
f
lambda_4/percentile_2/sub_1/yConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
В
lambda_4/percentile_2/sub_1Sublambda_4/percentile_2/ToDoublelambda_4/percentile_2/sub_1/y*
_output_shapes
: *
T0
y
lambda_4/percentile_2/mulMullambda_4/percentile_2/sub_1lambda_4/percentile_2/sub*
T0*
_output_shapes
: 
`
lambda_4/percentile_2/RoundRoundlambda_4/percentile_2/mul*
T0*
_output_shapes
: 
z
lambda_4/percentile_2/Shape_2Shapelambda_4/percentile_2/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_4/percentile_2/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w
-lambda_4/percentile_2/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
w
-lambda_4/percentile_2/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_4/percentile_2/strided_slice_2StridedSlicelambda_4/percentile_2/Shape_2+lambda_4/percentile_2/strided_slice_2/stack-lambda_4/percentile_2/strided_slice_2/stack_1-lambda_4/percentile_2/strided_slice_2/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
В
lambda_4/percentile_2/ToInt32Castlambda_4/percentile_2/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
_
lambda_4/percentile_2/sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
Й
lambda_4/percentile_2/sub_2Sub%lambda_4/percentile_2/strided_slice_2lambda_4/percentile_2/sub_2/y*
_output_shapes
: *
T0
У
+lambda_4/percentile_2/clip_by_value/MinimumMinimumlambda_4/percentile_2/ToInt32lambda_4/percentile_2/sub_2*
T0*
_output_shapes
: 
g
%lambda_4/percentile_2/clip_by_value/yConst*
value	B : *
dtype0*
_output_shapes
: 
£
#lambda_4/percentile_2/clip_by_valueMaximum+lambda_4/percentile_2/clip_by_value/Minimum%lambda_4/percentile_2/clip_by_value/y*
T0*
_output_shapes
: 
z
lambda_4/percentile_2/Shape_3Shapelambda_4/percentile_2/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_4/percentile_2/strided_slice_3/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_4/percentile_2/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_4/percentile_2/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_4/percentile_2/strided_slice_3StridedSlicelambda_4/percentile_2/Shape_3+lambda_4/percentile_2/strided_slice_3/stack-lambda_4/percentile_2/strided_slice_3/stack_1-lambda_4/percentile_2/strided_slice_3/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
ў
lambda_4/percentile_2/TopKV2TopKV2lambda_4/percentile_2/Reshape%lambda_4/percentile_2/strided_slice_3*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€*
sorted(*
T0
]
lambda_4/percentile_2/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
Г
lambda_4/percentile_2/addAdd#lambda_4/percentile_2/clip_by_valuelambda_4/percentile_2/add/y*
T0*
_output_shapes
: 
o
-lambda_4/percentile_2/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
+lambda_4/percentile_2/strided_slice_4/stackPack-lambda_4/percentile_2/strided_slice_4/stack/0#lambda_4/percentile_2/clip_by_value*
T0*

axis *
N*
_output_shapes
:
q
/lambda_4/percentile_2/strided_slice_4/stack_1/0Const*
dtype0*
_output_shapes
: *
value	B : 
ї
-lambda_4/percentile_2/strided_slice_4/stack_1Pack/lambda_4/percentile_2/strided_slice_4/stack_1/0lambda_4/percentile_2/add*
T0*

axis *
N*
_output_shapes
:
~
-lambda_4/percentile_2/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Б
%lambda_4/percentile_2/strided_slice_4StridedSlicelambda_4/percentile_2/TopKV2+lambda_4/percentile_2/strided_slice_4/stack-lambda_4/percentile_2/strided_slice_4/stack_1-lambda_4/percentile_2/strided_slice_4/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask*

begin_mask *
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€
f
$lambda_4/percentile_2/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
љ
 lambda_4/percentile_2/ExpandDims
ExpandDims%lambda_4/percentile_2/strided_slice_4$lambda_4/percentile_2/ExpandDims/dim*

Tdim0*
T0*+
_output_shapes
:€€€€€€€€€
h
&lambda_4/percentile_2/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
ј
"lambda_4/percentile_2/ExpandDims_1
ExpandDims lambda_4/percentile_2/ExpandDims&lambda_4/percentile_2/ExpandDims_1/dim*/
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
^
lambda_4/percentile_3/q/xConst*
valueB
 *ЪЩ«B*
dtype0*
_output_shapes
: 
z
lambda_4/percentile_3/qCastlambda_4/percentile_3/q/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
k
lambda_4/percentile_3/axisConst*
dtype0*
_output_shapes
:*
valueB"      
N
Flambda_4/percentile_3/assert_integer/statically_determined_was_integerNoOp
}
$lambda_4/percentile_3/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ѕ
lambda_4/percentile_3/transpose	Transposelambda_4/Placeholder$lambda_4/percentile_3/transpose/perm*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0
z
lambda_4/percentile_3/ShapeShapelambda_4/percentile_3/transpose*
_output_shapes
:*
T0*
out_type0
s
)lambda_4/percentile_3/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
u
+lambda_4/percentile_3/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
u
+lambda_4/percentile_3/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
л
#lambda_4/percentile_3/strided_sliceStridedSlicelambda_4/percentile_3/Shape)lambda_4/percentile_3/strided_slice/stack+lambda_4/percentile_3/strided_slice/stack_1+lambda_4/percentile_3/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask 
x
%lambda_4/percentile_3/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
c
!lambda_4/percentile_3/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
—
lambda_4/percentile_3/concatConcatV2#lambda_4/percentile_3/strided_slice%lambda_4/percentile_3/concat/values_1!lambda_4/percentile_3/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
і
lambda_4/percentile_3/ReshapeReshapelambda_4/percentile_3/transposelambda_4/percentile_3/concat*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
h
lambda_4/percentile_3/truediv/yConst*
valueB 2      Y@*
dtype0*
_output_shapes
: 
Г
lambda_4/percentile_3/truedivRealDivlambda_4/percentile_3/qlambda_4/percentile_3/truediv/y*
T0*
_output_shapes
: 
d
lambda_4/percentile_3/sub/xConst*
dtype0*
_output_shapes
: *
valueB 2      р?
}
lambda_4/percentile_3/subSublambda_4/percentile_3/sub/xlambda_4/percentile_3/truediv*
T0*
_output_shapes
: 
z
lambda_4/percentile_3/Shape_1Shapelambda_4/percentile_3/Reshape*
_output_shapes
:*
T0*
out_type0
~
+lambda_4/percentile_3/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_4/percentile_3/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
w
-lambda_4/percentile_3/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_4/percentile_3/strided_slice_1StridedSlicelambda_4/percentile_3/Shape_1+lambda_4/percentile_3/strided_slice_1/stack-lambda_4/percentile_3/strided_slice_1/stack_1-lambda_4/percentile_3/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Н
lambda_4/percentile_3/ToDoubleCast%lambda_4/percentile_3/strided_slice_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
f
lambda_4/percentile_3/sub_1/yConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
В
lambda_4/percentile_3/sub_1Sublambda_4/percentile_3/ToDoublelambda_4/percentile_3/sub_1/y*
T0*
_output_shapes
: 
y
lambda_4/percentile_3/mulMullambda_4/percentile_3/sub_1lambda_4/percentile_3/sub*
T0*
_output_shapes
: 
`
lambda_4/percentile_3/RoundRoundlambda_4/percentile_3/mul*
T0*
_output_shapes
: 
z
lambda_4/percentile_3/Shape_2Shapelambda_4/percentile_3/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_4/percentile_3/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w
-lambda_4/percentile_3/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
w
-lambda_4/percentile_3/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_4/percentile_3/strided_slice_2StridedSlicelambda_4/percentile_3/Shape_2+lambda_4/percentile_3/strided_slice_2/stack-lambda_4/percentile_3/strided_slice_2/stack_1-lambda_4/percentile_3/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
В
lambda_4/percentile_3/ToInt32Castlambda_4/percentile_3/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
_
lambda_4/percentile_3/sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Й
lambda_4/percentile_3/sub_2Sub%lambda_4/percentile_3/strided_slice_2lambda_4/percentile_3/sub_2/y*
T0*
_output_shapes
: 
У
+lambda_4/percentile_3/clip_by_value/MinimumMinimumlambda_4/percentile_3/ToInt32lambda_4/percentile_3/sub_2*
T0*
_output_shapes
: 
g
%lambda_4/percentile_3/clip_by_value/yConst*
value	B : *
dtype0*
_output_shapes
: 
£
#lambda_4/percentile_3/clip_by_valueMaximum+lambda_4/percentile_3/clip_by_value/Minimum%lambda_4/percentile_3/clip_by_value/y*
T0*
_output_shapes
: 
z
lambda_4/percentile_3/Shape_3Shapelambda_4/percentile_3/Reshape*
_output_shapes
:*
T0*
out_type0
~
+lambda_4/percentile_3/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w
-lambda_4/percentile_3/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_4/percentile_3/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_4/percentile_3/strided_slice_3StridedSlicelambda_4/percentile_3/Shape_3+lambda_4/percentile_3/strided_slice_3/stack-lambda_4/percentile_3/strided_slice_3/stack_1-lambda_4/percentile_3/strided_slice_3/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
ў
lambda_4/percentile_3/TopKV2TopKV2lambda_4/percentile_3/Reshape%lambda_4/percentile_3/strided_slice_3*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€*
sorted(
]
lambda_4/percentile_3/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
Г
lambda_4/percentile_3/addAdd#lambda_4/percentile_3/clip_by_valuelambda_4/percentile_3/add/y*
T0*
_output_shapes
: 
o
-lambda_4/percentile_3/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
+lambda_4/percentile_3/strided_slice_4/stackPack-lambda_4/percentile_3/strided_slice_4/stack/0#lambda_4/percentile_3/clip_by_value*
T0*

axis *
N*
_output_shapes
:
q
/lambda_4/percentile_3/strided_slice_4/stack_1/0Const*
dtype0*
_output_shapes
: *
value	B : 
ї
-lambda_4/percentile_3/strided_slice_4/stack_1Pack/lambda_4/percentile_3/strided_slice_4/stack_1/0lambda_4/percentile_3/add*
T0*

axis *
N*
_output_shapes
:
~
-lambda_4/percentile_3/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Б
%lambda_4/percentile_3/strided_slice_4StridedSlicelambda_4/percentile_3/TopKV2+lambda_4/percentile_3/strided_slice_4/stack-lambda_4/percentile_3/strided_slice_4/stack_1-lambda_4/percentile_3/strided_slice_4/stack_2*
shrink_axis_mask*
ellipsis_mask*

begin_mask *
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€*
T0*
Index0
f
$lambda_4/percentile_3/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
љ
 lambda_4/percentile_3/ExpandDims
ExpandDims%lambda_4/percentile_3/strided_slice_4$lambda_4/percentile_3/ExpandDims/dim*

Tdim0*
T0*+
_output_shapes
:€€€€€€€€€
h
&lambda_4/percentile_3/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
ј
"lambda_4/percentile_3/ExpandDims_1
ExpandDims lambda_4/percentile_3/ExpandDims&lambda_4/percentile_3/ExpandDims_1/dim*
T0*/
_output_shapes
:€€€€€€€€€*

Tdim0
Ы
lambda_4/sub_2Sublambda_4/Placeholder"lambda_4/percentile_2/ExpandDims_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ч
lambda_4/sub_3Sub"lambda_4/percentile_3/ExpandDims_1"lambda_4/percentile_2/ExpandDims_1*
T0*/
_output_shapes
:€€€€€€€€€
U
lambda_4/add_1/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
q
lambda_4/add_1Addlambda_4/sub_3lambda_4/add_1/y*
T0*/
_output_shapes
:€€€€€€€€€
Й
lambda_4/truediv_1RealDivlambda_4/sub_2lambda_4/add_1*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0
U
lambda_4/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *    
U
lambda_4/Const_3Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Э
 lambda_4/clip_by_value_1/MinimumMinimumlambda_4/truediv_1lambda_4/Const_3*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0
£
lambda_4/clip_by_value_1Maximum lambda_4/clip_by_value_1/Minimumlambda_4/Const_2*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
f
strided_slice_4/stackConst*
valueB"       *
dtype0*
_output_shapes
:
h
strided_slice_4/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
h
strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
ї
strided_slice_4StridedSliceconcatenate_3/concatstrided_slice_4/stackstrided_slice_4/stack_1strided_slice_4/stack_2*
ellipsis_mask*

begin_mask *
new_axis_mask *
end_mask*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Index0*
T0*
shrink_axis_mask 
_
strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:
a
strided_slice_5/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
a
strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ґ
strided_slice_5StridedSlicestrided_slice_4strided_slice_5/stackstrided_slice_5/stack_1strided_slice_5/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Y
lambda_5/percentile/q/xConst*
value	B : *
dtype0*
_output_shapes
: 
v
lambda_5/percentile/qCastlambda_5/percentile/q/x*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
i
lambda_5/percentile/axisConst*
valueB"      *
dtype0*
_output_shapes
:
L
Dlambda_5/percentile/assert_integer/statically_determined_was_integerNoOp
{
"lambda_5/percentile/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Є
lambda_5/percentile/transpose	Transposestrided_slice_5"lambda_5/percentile/transpose/perm*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0*
T0
v
lambda_5/percentile/ShapeShapelambda_5/percentile/transpose*
T0*
out_type0*
_output_shapes
:
q
'lambda_5/percentile/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
s
)lambda_5/percentile/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
s
)lambda_5/percentile/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
б
!lambda_5/percentile/strided_sliceStridedSlicelambda_5/percentile/Shape'lambda_5/percentile/strided_slice/stack)lambda_5/percentile/strided_slice/stack_1)lambda_5/percentile/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
v
#lambda_5/percentile/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
a
lambda_5/percentile/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
…
lambda_5/percentile/concatConcatV2!lambda_5/percentile/strided_slice#lambda_5/percentile/concat/values_1lambda_5/percentile/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ѓ
lambda_5/percentile/ReshapeReshapelambda_5/percentile/transposelambda_5/percentile/concat*
T0*
Tshape0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
f
lambda_5/percentile/truediv/yConst*
dtype0*
_output_shapes
: *
valueB 2      Y@
}
lambda_5/percentile/truedivRealDivlambda_5/percentile/qlambda_5/percentile/truediv/y*
T0*
_output_shapes
: 
b
lambda_5/percentile/sub/xConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
w
lambda_5/percentile/subSublambda_5/percentile/sub/xlambda_5/percentile/truediv*
_output_shapes
: *
T0
v
lambda_5/percentile/Shape_1Shapelambda_5/percentile/Reshape*
_output_shapes
:*
T0*
out_type0
|
)lambda_5/percentile/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
u
+lambda_5/percentile/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
u
+lambda_5/percentile/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
з
#lambda_5/percentile/strided_slice_1StridedSlicelambda_5/percentile/Shape_1)lambda_5/percentile/strided_slice_1/stack+lambda_5/percentile/strided_slice_1/stack_1+lambda_5/percentile/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Й
lambda_5/percentile/ToDoubleCast#lambda_5/percentile/strided_slice_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
d
lambda_5/percentile/sub_1/yConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
|
lambda_5/percentile/sub_1Sublambda_5/percentile/ToDoublelambda_5/percentile/sub_1/y*
T0*
_output_shapes
: 
s
lambda_5/percentile/mulMullambda_5/percentile/sub_1lambda_5/percentile/sub*
_output_shapes
: *
T0
\
lambda_5/percentile/RoundRoundlambda_5/percentile/mul*
T0*
_output_shapes
: 
v
lambda_5/percentile/Shape_2Shapelambda_5/percentile/Reshape*
_output_shapes
:*
T0*
out_type0
|
)lambda_5/percentile/strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
u
+lambda_5/percentile/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_5/percentile/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
з
#lambda_5/percentile/strided_slice_2StridedSlicelambda_5/percentile/Shape_2)lambda_5/percentile/strided_slice_2/stack+lambda_5/percentile/strided_slice_2/stack_1+lambda_5/percentile/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
~
lambda_5/percentile/ToInt32Castlambda_5/percentile/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
]
lambda_5/percentile/sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Г
lambda_5/percentile/sub_2Sub#lambda_5/percentile/strided_slice_2lambda_5/percentile/sub_2/y*
T0*
_output_shapes
: 
Н
)lambda_5/percentile/clip_by_value/MinimumMinimumlambda_5/percentile/ToInt32lambda_5/percentile/sub_2*
T0*
_output_shapes
: 
e
#lambda_5/percentile/clip_by_value/yConst*
dtype0*
_output_shapes
: *
value	B : 
Э
!lambda_5/percentile/clip_by_valueMaximum)lambda_5/percentile/clip_by_value/Minimum#lambda_5/percentile/clip_by_value/y*
_output_shapes
: *
T0
v
lambda_5/percentile/Shape_3Shapelambda_5/percentile/Reshape*
_output_shapes
:*
T0*
out_type0
|
)lambda_5/percentile/strided_slice_3/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
u
+lambda_5/percentile/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
u
+lambda_5/percentile/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
з
#lambda_5/percentile/strided_slice_3StridedSlicelambda_5/percentile/Shape_3)lambda_5/percentile/strided_slice_3/stack+lambda_5/percentile/strided_slice_3/stack_1+lambda_5/percentile/strided_slice_3/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
”
lambda_5/percentile/TopKV2TopKV2lambda_5/percentile/Reshape#lambda_5/percentile/strided_slice_3*
sorted(*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€
[
lambda_5/percentile/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
}
lambda_5/percentile/addAdd!lambda_5/percentile/clip_by_valuelambda_5/percentile/add/y*
T0*
_output_shapes
: 
m
+lambda_5/percentile/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
ї
)lambda_5/percentile/strided_slice_4/stackPack+lambda_5/percentile/strided_slice_4/stack/0!lambda_5/percentile/clip_by_value*
N*
_output_shapes
:*
T0*

axis 
o
-lambda_5/percentile/strided_slice_4/stack_1/0Const*
dtype0*
_output_shapes
: *
value	B : 
µ
+lambda_5/percentile/strided_slice_4/stack_1Pack-lambda_5/percentile/strided_slice_4/stack_1/0lambda_5/percentile/add*
T0*

axis *
N*
_output_shapes
:
|
+lambda_5/percentile/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
ч
#lambda_5/percentile/strided_slice_4StridedSlicelambda_5/percentile/TopKV2)lambda_5/percentile/strided_slice_4/stack+lambda_5/percentile/strided_slice_4/stack_1+lambda_5/percentile/strided_slice_4/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask*
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€
d
"lambda_5/percentile/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
Ј
lambda_5/percentile/ExpandDims
ExpandDims#lambda_5/percentile/strided_slice_4"lambda_5/percentile/ExpandDims/dim*+
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
f
$lambda_5/percentile/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B :
Ї
 lambda_5/percentile/ExpandDims_1
ExpandDimslambda_5/percentile/ExpandDims$lambda_5/percentile/ExpandDims_1/dim*
T0*/
_output_shapes
:€€€€€€€€€*

Tdim0
[
lambda_5/percentile_1/q/xConst*
value	B :d*
dtype0*
_output_shapes
: 
z
lambda_5/percentile_1/qCastlambda_5/percentile_1/q/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
k
lambda_5/percentile_1/axisConst*
dtype0*
_output_shapes
:*
valueB"      
N
Flambda_5/percentile_1/assert_integer/statically_determined_was_integerNoOp
}
$lambda_5/percentile_1/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Љ
lambda_5/percentile_1/transpose	Transposestrided_slice_5$lambda_5/percentile_1/transpose/perm*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0
z
lambda_5/percentile_1/ShapeShapelambda_5/percentile_1/transpose*
T0*
out_type0*
_output_shapes
:
s
)lambda_5/percentile_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
u
+lambda_5/percentile_1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
u
+lambda_5/percentile_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
л
#lambda_5/percentile_1/strided_sliceStridedSlicelambda_5/percentile_1/Shape)lambda_5/percentile_1/strided_slice/stack+lambda_5/percentile_1/strided_slice/stack_1+lambda_5/percentile_1/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
:*
T0*
Index0
x
%lambda_5/percentile_1/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
c
!lambda_5/percentile_1/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
—
lambda_5/percentile_1/concatConcatV2#lambda_5/percentile_1/strided_slice%lambda_5/percentile_1/concat/values_1!lambda_5/percentile_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
і
lambda_5/percentile_1/ReshapeReshapelambda_5/percentile_1/transposelambda_5/percentile_1/concat*
T0*
Tshape0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
h
lambda_5/percentile_1/truediv/yConst*
valueB 2      Y@*
dtype0*
_output_shapes
: 
Г
lambda_5/percentile_1/truedivRealDivlambda_5/percentile_1/qlambda_5/percentile_1/truediv/y*
T0*
_output_shapes
: 
d
lambda_5/percentile_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB 2      р?
}
lambda_5/percentile_1/subSublambda_5/percentile_1/sub/xlambda_5/percentile_1/truediv*
_output_shapes
: *
T0
z
lambda_5/percentile_1/Shape_1Shapelambda_5/percentile_1/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_5/percentile_1/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_5/percentile_1/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_5/percentile_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_5/percentile_1/strided_slice_1StridedSlicelambda_5/percentile_1/Shape_1+lambda_5/percentile_1/strided_slice_1/stack-lambda_5/percentile_1/strided_slice_1/stack_1-lambda_5/percentile_1/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Н
lambda_5/percentile_1/ToDoubleCast%lambda_5/percentile_1/strided_slice_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
f
lambda_5/percentile_1/sub_1/yConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
В
lambda_5/percentile_1/sub_1Sublambda_5/percentile_1/ToDoublelambda_5/percentile_1/sub_1/y*
T0*
_output_shapes
: 
y
lambda_5/percentile_1/mulMullambda_5/percentile_1/sub_1lambda_5/percentile_1/sub*
T0*
_output_shapes
: 
`
lambda_5/percentile_1/RoundRoundlambda_5/percentile_1/mul*
_output_shapes
: *
T0
z
lambda_5/percentile_1/Shape_2Shapelambda_5/percentile_1/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_5/percentile_1/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w
-lambda_5/percentile_1/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_5/percentile_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_5/percentile_1/strided_slice_2StridedSlicelambda_5/percentile_1/Shape_2+lambda_5/percentile_1/strided_slice_2/stack-lambda_5/percentile_1/strided_slice_2/stack_1-lambda_5/percentile_1/strided_slice_2/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
В
lambda_5/percentile_1/ToInt32Castlambda_5/percentile_1/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
_
lambda_5/percentile_1/sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
Й
lambda_5/percentile_1/sub_2Sub%lambda_5/percentile_1/strided_slice_2lambda_5/percentile_1/sub_2/y*
T0*
_output_shapes
: 
У
+lambda_5/percentile_1/clip_by_value/MinimumMinimumlambda_5/percentile_1/ToInt32lambda_5/percentile_1/sub_2*
T0*
_output_shapes
: 
g
%lambda_5/percentile_1/clip_by_value/yConst*
value	B : *
dtype0*
_output_shapes
: 
£
#lambda_5/percentile_1/clip_by_valueMaximum+lambda_5/percentile_1/clip_by_value/Minimum%lambda_5/percentile_1/clip_by_value/y*
T0*
_output_shapes
: 
z
lambda_5/percentile_1/Shape_3Shapelambda_5/percentile_1/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_5/percentile_1/strided_slice_3/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_5/percentile_1/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
w
-lambda_5/percentile_1/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_5/percentile_1/strided_slice_3StridedSlicelambda_5/percentile_1/Shape_3+lambda_5/percentile_1/strided_slice_3/stack-lambda_5/percentile_1/strided_slice_3/stack_1-lambda_5/percentile_1/strided_slice_3/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
ў
lambda_5/percentile_1/TopKV2TopKV2lambda_5/percentile_1/Reshape%lambda_5/percentile_1/strided_slice_3*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€*
sorted(
]
lambda_5/percentile_1/add/yConst*
dtype0*
_output_shapes
: *
value	B :
Г
lambda_5/percentile_1/addAdd#lambda_5/percentile_1/clip_by_valuelambda_5/percentile_1/add/y*
T0*
_output_shapes
: 
o
-lambda_5/percentile_1/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
+lambda_5/percentile_1/strided_slice_4/stackPack-lambda_5/percentile_1/strided_slice_4/stack/0#lambda_5/percentile_1/clip_by_value*
T0*

axis *
N*
_output_shapes
:
q
/lambda_5/percentile_1/strided_slice_4/stack_1/0Const*
dtype0*
_output_shapes
: *
value	B : 
ї
-lambda_5/percentile_1/strided_slice_4/stack_1Pack/lambda_5/percentile_1/strided_slice_4/stack_1/0lambda_5/percentile_1/add*
T0*

axis *
N*
_output_shapes
:
~
-lambda_5/percentile_1/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Б
%lambda_5/percentile_1/strided_slice_4StridedSlicelambda_5/percentile_1/TopKV2+lambda_5/percentile_1/strided_slice_4/stack-lambda_5/percentile_1/strided_slice_4/stack_1-lambda_5/percentile_1/strided_slice_4/stack_2*
shrink_axis_mask*
ellipsis_mask*

begin_mask *
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€*
T0*
Index0
f
$lambda_5/percentile_1/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
љ
 lambda_5/percentile_1/ExpandDims
ExpandDims%lambda_5/percentile_1/strided_slice_4$lambda_5/percentile_1/ExpandDims/dim*

Tdim0*
T0*+
_output_shapes
:€€€€€€€€€
h
&lambda_5/percentile_1/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
ј
"lambda_5/percentile_1/ExpandDims_1
ExpandDims lambda_5/percentile_1/ExpandDims&lambda_5/percentile_1/ExpandDims_1/dim*

Tdim0*
T0*/
_output_shapes
:€€€€€€€€€
Т
lambda_5/subSubstrided_slice_5 lambda_5/percentile/ExpandDims_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Х
lambda_5/sub_1Sub"lambda_5/percentile_1/ExpandDims_1 lambda_5/percentile/ExpandDims_1*
T0*/
_output_shapes
:€€€€€€€€€
S
lambda_5/add/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
m
lambda_5/addAddlambda_5/sub_1lambda_5/add/y*/
_output_shapes
:€€€€€€€€€*
T0
Г
lambda_5/truedivRealDivlambda_5/sublambda_5/add*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
S
lambda_5/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
U
lambda_5/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Щ
lambda_5/clip_by_value/MinimumMinimumlambda_5/truedivlambda_5/Const_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Э
lambda_5/clip_by_valueMaximumlambda_5/clip_by_value/Minimumlambda_5/Const*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ђ
lambda_5/PlaceholderPlaceholder*
dtype0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*6
shape-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
[
lambda_5/percentile_2/q/xConst*
value	B : *
dtype0*
_output_shapes
: 
z
lambda_5/percentile_2/qCastlambda_5/percentile_2/q/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
k
lambda_5/percentile_2/axisConst*
dtype0*
_output_shapes
:*
valueB"      
N
Flambda_5/percentile_2/assert_integer/statically_determined_was_integerNoOp
}
$lambda_5/percentile_2/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ѕ
lambda_5/percentile_2/transpose	Transposelambda_5/Placeholder$lambda_5/percentile_2/transpose/perm*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0
z
lambda_5/percentile_2/ShapeShapelambda_5/percentile_2/transpose*
T0*
out_type0*
_output_shapes
:
s
)lambda_5/percentile_2/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_5/percentile_2/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+lambda_5/percentile_2/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
л
#lambda_5/percentile_2/strided_sliceStridedSlicelambda_5/percentile_2/Shape)lambda_5/percentile_2/strided_slice/stack+lambda_5/percentile_2/strided_slice/stack_1+lambda_5/percentile_2/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
:
x
%lambda_5/percentile_2/concat/values_1Const*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
c
!lambda_5/percentile_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
—
lambda_5/percentile_2/concatConcatV2#lambda_5/percentile_2/strided_slice%lambda_5/percentile_2/concat/values_1!lambda_5/percentile_2/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
і
lambda_5/percentile_2/ReshapeReshapelambda_5/percentile_2/transposelambda_5/percentile_2/concat*
T0*
Tshape0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
h
lambda_5/percentile_2/truediv/yConst*
valueB 2      Y@*
dtype0*
_output_shapes
: 
Г
lambda_5/percentile_2/truedivRealDivlambda_5/percentile_2/qlambda_5/percentile_2/truediv/y*
_output_shapes
: *
T0
d
lambda_5/percentile_2/sub/xConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
}
lambda_5/percentile_2/subSublambda_5/percentile_2/sub/xlambda_5/percentile_2/truediv*
T0*
_output_shapes
: 
z
lambda_5/percentile_2/Shape_1Shapelambda_5/percentile_2/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_5/percentile_2/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_5/percentile_2/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
w
-lambda_5/percentile_2/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
с
%lambda_5/percentile_2/strided_slice_1StridedSlicelambda_5/percentile_2/Shape_1+lambda_5/percentile_2/strided_slice_1/stack-lambda_5/percentile_2/strided_slice_1/stack_1-lambda_5/percentile_2/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Н
lambda_5/percentile_2/ToDoubleCast%lambda_5/percentile_2/strided_slice_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
f
lambda_5/percentile_2/sub_1/yConst*
dtype0*
_output_shapes
: *
valueB 2      р?
В
lambda_5/percentile_2/sub_1Sublambda_5/percentile_2/ToDoublelambda_5/percentile_2/sub_1/y*
_output_shapes
: *
T0
y
lambda_5/percentile_2/mulMullambda_5/percentile_2/sub_1lambda_5/percentile_2/sub*
T0*
_output_shapes
: 
`
lambda_5/percentile_2/RoundRoundlambda_5/percentile_2/mul*
_output_shapes
: *
T0
z
lambda_5/percentile_2/Shape_2Shapelambda_5/percentile_2/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_5/percentile_2/strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_5/percentile_2/strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
w
-lambda_5/percentile_2/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_5/percentile_2/strided_slice_2StridedSlicelambda_5/percentile_2/Shape_2+lambda_5/percentile_2/strided_slice_2/stack-lambda_5/percentile_2/strided_slice_2/stack_1-lambda_5/percentile_2/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
В
lambda_5/percentile_2/ToInt32Castlambda_5/percentile_2/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
_
lambda_5/percentile_2/sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Й
lambda_5/percentile_2/sub_2Sub%lambda_5/percentile_2/strided_slice_2lambda_5/percentile_2/sub_2/y*
_output_shapes
: *
T0
У
+lambda_5/percentile_2/clip_by_value/MinimumMinimumlambda_5/percentile_2/ToInt32lambda_5/percentile_2/sub_2*
_output_shapes
: *
T0
g
%lambda_5/percentile_2/clip_by_value/yConst*
value	B : *
dtype0*
_output_shapes
: 
£
#lambda_5/percentile_2/clip_by_valueMaximum+lambda_5/percentile_2/clip_by_value/Minimum%lambda_5/percentile_2/clip_by_value/y*
_output_shapes
: *
T0
z
lambda_5/percentile_2/Shape_3Shapelambda_5/percentile_2/Reshape*
_output_shapes
:*
T0*
out_type0
~
+lambda_5/percentile_2/strided_slice_3/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_5/percentile_2/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_5/percentile_2/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_5/percentile_2/strided_slice_3StridedSlicelambda_5/percentile_2/Shape_3+lambda_5/percentile_2/strided_slice_3/stack-lambda_5/percentile_2/strided_slice_3/stack_1-lambda_5/percentile_2/strided_slice_3/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
ў
lambda_5/percentile_2/TopKV2TopKV2lambda_5/percentile_2/Reshape%lambda_5/percentile_2/strided_slice_3*
sorted(*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€
]
lambda_5/percentile_2/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
Г
lambda_5/percentile_2/addAdd#lambda_5/percentile_2/clip_by_valuelambda_5/percentile_2/add/y*
T0*
_output_shapes
: 
o
-lambda_5/percentile_2/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
+lambda_5/percentile_2/strided_slice_4/stackPack-lambda_5/percentile_2/strided_slice_4/stack/0#lambda_5/percentile_2/clip_by_value*
T0*

axis *
N*
_output_shapes
:
q
/lambda_5/percentile_2/strided_slice_4/stack_1/0Const*
dtype0*
_output_shapes
: *
value	B : 
ї
-lambda_5/percentile_2/strided_slice_4/stack_1Pack/lambda_5/percentile_2/strided_slice_4/stack_1/0lambda_5/percentile_2/add*
T0*

axis *
N*
_output_shapes
:
~
-lambda_5/percentile_2/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
Б
%lambda_5/percentile_2/strided_slice_4StridedSlicelambda_5/percentile_2/TopKV2+lambda_5/percentile_2/strided_slice_4/stack-lambda_5/percentile_2/strided_slice_4/stack_1-lambda_5/percentile_2/strided_slice_4/stack_2*
shrink_axis_mask*
ellipsis_mask*

begin_mask *
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€*
Index0*
T0
f
$lambda_5/percentile_2/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
љ
 lambda_5/percentile_2/ExpandDims
ExpandDims%lambda_5/percentile_2/strided_slice_4$lambda_5/percentile_2/ExpandDims/dim*

Tdim0*
T0*+
_output_shapes
:€€€€€€€€€
h
&lambda_5/percentile_2/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
ј
"lambda_5/percentile_2/ExpandDims_1
ExpandDims lambda_5/percentile_2/ExpandDims&lambda_5/percentile_2/ExpandDims_1/dim*

Tdim0*
T0*/
_output_shapes
:€€€€€€€€€
[
lambda_5/percentile_3/q/xConst*
value	B :d*
dtype0*
_output_shapes
: 
z
lambda_5/percentile_3/qCastlambda_5/percentile_3/q/x*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
k
lambda_5/percentile_3/axisConst*
valueB"      *
dtype0*
_output_shapes
:
N
Flambda_5/percentile_3/assert_integer/statically_determined_was_integerNoOp
}
$lambda_5/percentile_3/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ѕ
lambda_5/percentile_3/transpose	Transposelambda_5/Placeholder$lambda_5/percentile_3/transpose/perm*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
Tperm0
z
lambda_5/percentile_3/ShapeShapelambda_5/percentile_3/transpose*
T0*
out_type0*
_output_shapes
:
s
)lambda_5/percentile_3/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
u
+lambda_5/percentile_3/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
u
+lambda_5/percentile_3/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
л
#lambda_5/percentile_3/strided_sliceStridedSlicelambda_5/percentile_3/Shape)lambda_5/percentile_3/strided_slice/stack+lambda_5/percentile_3/strided_slice/stack_1+lambda_5/percentile_3/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
x
%lambda_5/percentile_3/concat/values_1Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
c
!lambda_5/percentile_3/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
—
lambda_5/percentile_3/concatConcatV2#lambda_5/percentile_3/strided_slice%lambda_5/percentile_3/concat/values_1!lambda_5/percentile_3/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
і
lambda_5/percentile_3/ReshapeReshapelambda_5/percentile_3/transposelambda_5/percentile_3/concat*
T0*
Tshape0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
h
lambda_5/percentile_3/truediv/yConst*
dtype0*
_output_shapes
: *
valueB 2      Y@
Г
lambda_5/percentile_3/truedivRealDivlambda_5/percentile_3/qlambda_5/percentile_3/truediv/y*
_output_shapes
: *
T0
d
lambda_5/percentile_3/sub/xConst*
valueB 2      р?*
dtype0*
_output_shapes
: 
}
lambda_5/percentile_3/subSublambda_5/percentile_3/sub/xlambda_5/percentile_3/truediv*
T0*
_output_shapes
: 
z
lambda_5/percentile_3/Shape_1Shapelambda_5/percentile_3/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_5/percentile_3/strided_slice_1/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_5/percentile_3/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_5/percentile_3/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_5/percentile_3/strided_slice_1StridedSlicelambda_5/percentile_3/Shape_1+lambda_5/percentile_3/strided_slice_1/stack-lambda_5/percentile_3/strided_slice_1/stack_1-lambda_5/percentile_3/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Н
lambda_5/percentile_3/ToDoubleCast%lambda_5/percentile_3/strided_slice_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
f
lambda_5/percentile_3/sub_1/yConst*
dtype0*
_output_shapes
: *
valueB 2      р?
В
lambda_5/percentile_3/sub_1Sublambda_5/percentile_3/ToDoublelambda_5/percentile_3/sub_1/y*
_output_shapes
: *
T0
y
lambda_5/percentile_3/mulMullambda_5/percentile_3/sub_1lambda_5/percentile_3/sub*
_output_shapes
: *
T0
`
lambda_5/percentile_3/RoundRoundlambda_5/percentile_3/mul*
T0*
_output_shapes
: 
z
lambda_5/percentile_3/Shape_2Shapelambda_5/percentile_3/Reshape*
_output_shapes
:*
T0*
out_type0
~
+lambda_5/percentile_3/strided_slice_2/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_5/percentile_3/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
w
-lambda_5/percentile_3/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_5/percentile_3/strided_slice_2StridedSlicelambda_5/percentile_3/Shape_2+lambda_5/percentile_3/strided_slice_2/stack-lambda_5/percentile_3/strided_slice_2/stack_1-lambda_5/percentile_3/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
В
lambda_5/percentile_3/ToInt32Castlambda_5/percentile_3/Round*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
_
lambda_5/percentile_3/sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Й
lambda_5/percentile_3/sub_2Sub%lambda_5/percentile_3/strided_slice_2lambda_5/percentile_3/sub_2/y*
_output_shapes
: *
T0
У
+lambda_5/percentile_3/clip_by_value/MinimumMinimumlambda_5/percentile_3/ToInt32lambda_5/percentile_3/sub_2*
T0*
_output_shapes
: 
g
%lambda_5/percentile_3/clip_by_value/yConst*
value	B : *
dtype0*
_output_shapes
: 
£
#lambda_5/percentile_3/clip_by_valueMaximum+lambda_5/percentile_3/clip_by_value/Minimum%lambda_5/percentile_3/clip_by_value/y*
_output_shapes
: *
T0
z
lambda_5/percentile_3/Shape_3Shapelambda_5/percentile_3/Reshape*
T0*
out_type0*
_output_shapes
:
~
+lambda_5/percentile_3/strided_slice_3/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
-lambda_5/percentile_3/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
w
-lambda_5/percentile_3/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
с
%lambda_5/percentile_3/strided_slice_3StridedSlicelambda_5/percentile_3/Shape_3+lambda_5/percentile_3/strided_slice_3/stack-lambda_5/percentile_3/strided_slice_3/stack_1-lambda_5/percentile_3/strided_slice_3/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
ў
lambda_5/percentile_3/TopKV2TopKV2lambda_5/percentile_3/Reshape%lambda_5/percentile_3/strided_slice_3*
sorted(*
T0*T
_output_shapesB
@:€€€€€€€€€€€€€€€€€€:€€€€€€€€€€€€€€€€€€
]
lambda_5/percentile_3/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
Г
lambda_5/percentile_3/addAdd#lambda_5/percentile_3/clip_by_valuelambda_5/percentile_3/add/y*
_output_shapes
: *
T0
o
-lambda_5/percentile_3/strided_slice_4/stack/0Const*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
+lambda_5/percentile_3/strided_slice_4/stackPack-lambda_5/percentile_3/strided_slice_4/stack/0#lambda_5/percentile_3/clip_by_value*
T0*

axis *
N*
_output_shapes
:
q
/lambda_5/percentile_3/strided_slice_4/stack_1/0Const*
value	B : *
dtype0*
_output_shapes
: 
ї
-lambda_5/percentile_3/strided_slice_4/stack_1Pack/lambda_5/percentile_3/strided_slice_4/stack_1/0lambda_5/percentile_3/add*
N*
_output_shapes
:*
T0*

axis 
~
-lambda_5/percentile_3/strided_slice_4/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Б
%lambda_5/percentile_3/strided_slice_4StridedSlicelambda_5/percentile_3/TopKV2+lambda_5/percentile_3/strided_slice_4/stack-lambda_5/percentile_3/strided_slice_4/stack_1-lambda_5/percentile_3/strided_slice_4/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask*
new_axis_mask *
end_mask *'
_output_shapes
:€€€€€€€€€*
T0*
Index0
f
$lambda_5/percentile_3/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
љ
 lambda_5/percentile_3/ExpandDims
ExpandDims%lambda_5/percentile_3/strided_slice_4$lambda_5/percentile_3/ExpandDims/dim*

Tdim0*
T0*+
_output_shapes
:€€€€€€€€€
h
&lambda_5/percentile_3/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
ј
"lambda_5/percentile_3/ExpandDims_1
ExpandDims lambda_5/percentile_3/ExpandDims&lambda_5/percentile_3/ExpandDims_1/dim*

Tdim0*
T0*/
_output_shapes
:€€€€€€€€€
Ы
lambda_5/sub_2Sublambda_5/Placeholder"lambda_5/percentile_2/ExpandDims_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ч
lambda_5/sub_3Sub"lambda_5/percentile_3/ExpandDims_1"lambda_5/percentile_2/ExpandDims_1*/
_output_shapes
:€€€€€€€€€*
T0
U
lambda_5/add_1/yConst*
valueB
 *Хњ÷3*
dtype0*
_output_shapes
: 
q
lambda_5/add_1Addlambda_5/sub_3lambda_5/add_1/y*
T0*/
_output_shapes
:€€€€€€€€€
Й
lambda_5/truediv_1RealDivlambda_5/sub_2lambda_5/add_1*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
U
lambda_5/Const_2Const*
valueB
 *    *
dtype0*
_output_shapes
: 
U
lambda_5/Const_3Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
Э
 lambda_5/clip_by_value_1/MinimumMinimumlambda_5/truediv_1lambda_5/Const_3*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
£
lambda_5/clip_by_value_1Maximum lambda_5/clip_by_value_1/Minimumlambda_5/Const_2*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
c
net_output_mean/tagConst* 
valueB Bnet_output_mean*
dtype0*
_output_shapes
: 
†
net_output_meanImageSummarynet_output_mean/taglambda_4/clip_by_value*
_output_shapes
: *

max_images*
T0*
	bad_colorB:€  €
e
net_output_scale/tagConst*!
valueB Bnet_output_scale*
dtype0*
_output_shapes
: 
Ґ
net_output_scaleImageSummarynet_output_scale/taglambda_5/clip_by_value*

max_images*
T0*
	bad_colorB:€  €*
_output_shapes
: 
Д
merged/Merge/MergeSummaryMergeSummary	net_input
net_targetnet_output_meannet_output_scale*
N*
_output_shapes
: 
N
Placeholder_1Placeholder*
shape: *
dtype0*
_output_shapes
: 
О
AssignAssignAdam/lrPlaceholder_1*
use_locking( *
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: 
n
Placeholder_2Placeholder*
dtype0*&
_output_shapes
: *
shape: 
¬
Assign_1Assigndown_level_0_no_0/kernelPlaceholder_2*
use_locking( *
T0*+
_class!
loc:@down_level_0_no_0/kernel*
validate_shape(*&
_output_shapes
: 
V
Placeholder_3Placeholder*
dtype0*
_output_shapes
: *
shape: 
≤
Assign_2Assigndown_level_0_no_0/biasPlaceholder_3*
T0*)
_class
loc:@down_level_0_no_0/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
n
Placeholder_4Placeholder*
dtype0*&
_output_shapes
:  *
shape:  
¬
Assign_3Assigndown_level_0_no_1/kernelPlaceholder_4*
use_locking( *
T0*+
_class!
loc:@down_level_0_no_1/kernel*
validate_shape(*&
_output_shapes
:  
V
Placeholder_5Placeholder*
dtype0*
_output_shapes
: *
shape: 
≤
Assign_4Assigndown_level_0_no_1/biasPlaceholder_5*
use_locking( *
T0*)
_class
loc:@down_level_0_no_1/bias*
validate_shape(*
_output_shapes
: 
n
Placeholder_6Placeholder*
dtype0*&
_output_shapes
: @*
shape: @
¬
Assign_5Assigndown_level_1_no_0/kernelPlaceholder_6*
use_locking( *
T0*+
_class!
loc:@down_level_1_no_0/kernel*
validate_shape(*&
_output_shapes
: @
V
Placeholder_7Placeholder*
shape:@*
dtype0*
_output_shapes
:@
≤
Assign_6Assigndown_level_1_no_0/biasPlaceholder_7*
validate_shape(*
_output_shapes
:@*
use_locking( *
T0*)
_class
loc:@down_level_1_no_0/bias
n
Placeholder_8Placeholder*
dtype0*&
_output_shapes
:@@*
shape:@@
¬
Assign_7Assigndown_level_1_no_1/kernelPlaceholder_8*
use_locking( *
T0*+
_class!
loc:@down_level_1_no_1/kernel*
validate_shape(*&
_output_shapes
:@@
V
Placeholder_9Placeholder*
dtype0*
_output_shapes
:@*
shape:@
≤
Assign_8Assigndown_level_1_no_1/biasPlaceholder_9*
T0*)
_class
loc:@down_level_1_no_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking( 
q
Placeholder_10Placeholder*
dtype0*'
_output_shapes
:@А*
shape:@А
≤
Assign_9Assignmiddle_0/kernelPlaceholder_10*
validate_shape(*'
_output_shapes
:@А*
use_locking( *
T0*"
_class
loc:@middle_0/kernel
Y
Placeholder_11Placeholder*
dtype0*
_output_shapes	
:А*
shape:А
£
	Assign_10Assignmiddle_0/biasPlaceholder_11*
use_locking( *
T0* 
_class
loc:@middle_0/bias*
validate_shape(*
_output_shapes	
:А
q
Placeholder_12Placeholder*
shape:А@*
dtype0*'
_output_shapes
:А@
≥
	Assign_11Assignmiddle_2/kernelPlaceholder_12*
use_locking( *
T0*"
_class
loc:@middle_2/kernel*
validate_shape(*'
_output_shapes
:А@
W
Placeholder_13Placeholder*
shape:@*
dtype0*
_output_shapes
:@
Ґ
	Assign_12Assignmiddle_2/biasPlaceholder_13*
use_locking( *
T0* 
_class
loc:@middle_2/bias*
validate_shape(*
_output_shapes
:@
q
Placeholder_14Placeholder*
dtype0*'
_output_shapes
:А@*
shape:А@
Ѕ
	Assign_13Assignup_level_1_no_0/kernelPlaceholder_14*
use_locking( *
T0*)
_class
loc:@up_level_1_no_0/kernel*
validate_shape(*'
_output_shapes
:А@
W
Placeholder_15Placeholder*
dtype0*
_output_shapes
:@*
shape:@
∞
	Assign_14Assignup_level_1_no_0/biasPlaceholder_15*
T0*'
_class
loc:@up_level_1_no_0/bias*
validate_shape(*
_output_shapes
:@*
use_locking( 
o
Placeholder_16Placeholder*
shape:@ *
dtype0*&
_output_shapes
:@ 
ј
	Assign_15Assignup_level_1_no_2/kernelPlaceholder_16*
validate_shape(*&
_output_shapes
:@ *
use_locking( *
T0*)
_class
loc:@up_level_1_no_2/kernel
W
Placeholder_17Placeholder*
shape: *
dtype0*
_output_shapes
: 
∞
	Assign_16Assignup_level_1_no_2/biasPlaceholder_17*
use_locking( *
T0*'
_class
loc:@up_level_1_no_2/bias*
validate_shape(*
_output_shapes
: 
o
Placeholder_18Placeholder*
dtype0*&
_output_shapes
:@ *
shape:@ 
ј
	Assign_17Assignup_level_0_no_0/kernelPlaceholder_18*
validate_shape(*&
_output_shapes
:@ *
use_locking( *
T0*)
_class
loc:@up_level_0_no_0/kernel
W
Placeholder_19Placeholder*
dtype0*
_output_shapes
: *
shape: 
∞
	Assign_18Assignup_level_0_no_0/biasPlaceholder_19*
use_locking( *
T0*'
_class
loc:@up_level_0_no_0/bias*
validate_shape(*
_output_shapes
: 
o
Placeholder_20Placeholder*
dtype0*&
_output_shapes
:  *
shape:  
ј
	Assign_19Assignup_level_0_no_2/kernelPlaceholder_20*
use_locking( *
T0*)
_class
loc:@up_level_0_no_2/kernel*
validate_shape(*&
_output_shapes
:  
W
Placeholder_21Placeholder*
dtype0*
_output_shapes
: *
shape: 
∞
	Assign_20Assignup_level_0_no_2/biasPlaceholder_21*
use_locking( *
T0*'
_class
loc:@up_level_0_no_2/bias*
validate_shape(*
_output_shapes
: 
o
Placeholder_22Placeholder*
dtype0*&
_output_shapes
: *
shape: 
≤
	Assign_21Assignconv2d_1/kernelPlaceholder_22*
use_locking( *
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
: 
W
Placeholder_23Placeholder*
shape:*
dtype0*
_output_shapes
:
Ґ
	Assign_22Assignconv2d_1/biasPlaceholder_23*
use_locking( *
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:
o
Placeholder_24Placeholder*
dtype0*&
_output_shapes
: *
shape: 
≤
	Assign_23Assignconv2d_2/kernelPlaceholder_24*
use_locking( *
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
: 
W
Placeholder_25Placeholder*
dtype0*
_output_shapes
:*
shape:
Ґ
	Assign_24Assignconv2d_2/biasPlaceholder_25*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:*
use_locking( 
+
group_deps_1NoOp^concatenate_3/concat
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_79d91bfb2ee74f0a9b45446395f1410b/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
љ
save/SaveV2/tensor_namesConst"/device:CPU:0*б
value„B‘eBAdam/beta_1BAdam/beta_2B
Adam/decayBAdam/iterationsBAdam/lrBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBdown_level_0_no_0/biasBdown_level_0_no_0/kernelBdown_level_0_no_1/biasBdown_level_0_no_1/kernelBdown_level_1_no_0/biasBdown_level_1_no_0/kernelBdown_level_1_no_1/biasBdown_level_1_no_1/kernelBmiddle_0/biasBmiddle_0/kernelBmiddle_2/biasBmiddle_2/kernelBtraining/Adam/VariableBtraining/Adam/Variable_1Btraining/Adam/Variable_10Btraining/Adam/Variable_11Btraining/Adam/Variable_12Btraining/Adam/Variable_13Btraining/Adam/Variable_14Btraining/Adam/Variable_15Btraining/Adam/Variable_16Btraining/Adam/Variable_17Btraining/Adam/Variable_18Btraining/Adam/Variable_19Btraining/Adam/Variable_2Btraining/Adam/Variable_20Btraining/Adam/Variable_21Btraining/Adam/Variable_22Btraining/Adam/Variable_23Btraining/Adam/Variable_24Btraining/Adam/Variable_25Btraining/Adam/Variable_26Btraining/Adam/Variable_27Btraining/Adam/Variable_28Btraining/Adam/Variable_29Btraining/Adam/Variable_3Btraining/Adam/Variable_30Btraining/Adam/Variable_31Btraining/Adam/Variable_32Btraining/Adam/Variable_33Btraining/Adam/Variable_34Btraining/Adam/Variable_35Btraining/Adam/Variable_36Btraining/Adam/Variable_37Btraining/Adam/Variable_38Btraining/Adam/Variable_39Btraining/Adam/Variable_4Btraining/Adam/Variable_40Btraining/Adam/Variable_41Btraining/Adam/Variable_42Btraining/Adam/Variable_43Btraining/Adam/Variable_44Btraining/Adam/Variable_45Btraining/Adam/Variable_46Btraining/Adam/Variable_47Btraining/Adam/Variable_48Btraining/Adam/Variable_49Btraining/Adam/Variable_5Btraining/Adam/Variable_50Btraining/Adam/Variable_51Btraining/Adam/Variable_52Btraining/Adam/Variable_53Btraining/Adam/Variable_54Btraining/Adam/Variable_55Btraining/Adam/Variable_56Btraining/Adam/Variable_57Btraining/Adam/Variable_58Btraining/Adam/Variable_59Btraining/Adam/Variable_6Btraining/Adam/Variable_60Btraining/Adam/Variable_61Btraining/Adam/Variable_62Btraining/Adam/Variable_63Btraining/Adam/Variable_64Btraining/Adam/Variable_65Btraining/Adam/Variable_66Btraining/Adam/Variable_67Btraining/Adam/Variable_68Btraining/Adam/Variable_69Btraining/Adam/Variable_7Btraining/Adam/Variable_70Btraining/Adam/Variable_71Btraining/Adam/Variable_8Btraining/Adam/Variable_9Bup_level_0_no_0/biasBup_level_0_no_0/kernelBup_level_0_no_2/biasBup_level_0_no_2/kernelBup_level_1_no_0/biasBup_level_1_no_0/kernelBup_level_1_no_2/biasBup_level_1_no_2/kernel*
dtype0*
_output_shapes
:e
њ
save/SaveV2/shape_and_slicesConst"/device:CPU:0*я
value’B“eB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:e
≥
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesAdam/beta_1Adam/beta_2
Adam/decayAdam/iterationsAdam/lrconv2d_1/biasconv2d_1/kernelconv2d_2/biasconv2d_2/kerneldown_level_0_no_0/biasdown_level_0_no_0/kerneldown_level_0_no_1/biasdown_level_0_no_1/kerneldown_level_1_no_0/biasdown_level_1_no_0/kerneldown_level_1_no_1/biasdown_level_1_no_1/kernelmiddle_0/biasmiddle_0/kernelmiddle_2/biasmiddle_2/kerneltraining/Adam/Variabletraining/Adam/Variable_1training/Adam/Variable_10training/Adam/Variable_11training/Adam/Variable_12training/Adam/Variable_13training/Adam/Variable_14training/Adam/Variable_15training/Adam/Variable_16training/Adam/Variable_17training/Adam/Variable_18training/Adam/Variable_19training/Adam/Variable_2training/Adam/Variable_20training/Adam/Variable_21training/Adam/Variable_22training/Adam/Variable_23training/Adam/Variable_24training/Adam/Variable_25training/Adam/Variable_26training/Adam/Variable_27training/Adam/Variable_28training/Adam/Variable_29training/Adam/Variable_3training/Adam/Variable_30training/Adam/Variable_31training/Adam/Variable_32training/Adam/Variable_33training/Adam/Variable_34training/Adam/Variable_35training/Adam/Variable_36training/Adam/Variable_37training/Adam/Variable_38training/Adam/Variable_39training/Adam/Variable_4training/Adam/Variable_40training/Adam/Variable_41training/Adam/Variable_42training/Adam/Variable_43training/Adam/Variable_44training/Adam/Variable_45training/Adam/Variable_46training/Adam/Variable_47training/Adam/Variable_48training/Adam/Variable_49training/Adam/Variable_5training/Adam/Variable_50training/Adam/Variable_51training/Adam/Variable_52training/Adam/Variable_53training/Adam/Variable_54training/Adam/Variable_55training/Adam/Variable_56training/Adam/Variable_57training/Adam/Variable_58training/Adam/Variable_59training/Adam/Variable_6training/Adam/Variable_60training/Adam/Variable_61training/Adam/Variable_62training/Adam/Variable_63training/Adam/Variable_64training/Adam/Variable_65training/Adam/Variable_66training/Adam/Variable_67training/Adam/Variable_68training/Adam/Variable_69training/Adam/Variable_7training/Adam/Variable_70training/Adam/Variable_71training/Adam/Variable_8training/Adam/Variable_9up_level_0_no_0/biasup_level_0_no_0/kernelup_level_0_no_2/biasup_level_0_no_2/kernelup_level_1_no_0/biasup_level_1_no_0/kernelup_level_1_no_2/biasup_level_1_no_2/kernel"/device:CPU:0*s
dtypesi
g2e	
†
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
ђ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
М
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
Й
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
ј
save/RestoreV2/tensor_namesConst"/device:CPU:0*б
value„B‘eBAdam/beta_1BAdam/beta_2B
Adam/decayBAdam/iterationsBAdam/lrBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBdown_level_0_no_0/biasBdown_level_0_no_0/kernelBdown_level_0_no_1/biasBdown_level_0_no_1/kernelBdown_level_1_no_0/biasBdown_level_1_no_0/kernelBdown_level_1_no_1/biasBdown_level_1_no_1/kernelBmiddle_0/biasBmiddle_0/kernelBmiddle_2/biasBmiddle_2/kernelBtraining/Adam/VariableBtraining/Adam/Variable_1Btraining/Adam/Variable_10Btraining/Adam/Variable_11Btraining/Adam/Variable_12Btraining/Adam/Variable_13Btraining/Adam/Variable_14Btraining/Adam/Variable_15Btraining/Adam/Variable_16Btraining/Adam/Variable_17Btraining/Adam/Variable_18Btraining/Adam/Variable_19Btraining/Adam/Variable_2Btraining/Adam/Variable_20Btraining/Adam/Variable_21Btraining/Adam/Variable_22Btraining/Adam/Variable_23Btraining/Adam/Variable_24Btraining/Adam/Variable_25Btraining/Adam/Variable_26Btraining/Adam/Variable_27Btraining/Adam/Variable_28Btraining/Adam/Variable_29Btraining/Adam/Variable_3Btraining/Adam/Variable_30Btraining/Adam/Variable_31Btraining/Adam/Variable_32Btraining/Adam/Variable_33Btraining/Adam/Variable_34Btraining/Adam/Variable_35Btraining/Adam/Variable_36Btraining/Adam/Variable_37Btraining/Adam/Variable_38Btraining/Adam/Variable_39Btraining/Adam/Variable_4Btraining/Adam/Variable_40Btraining/Adam/Variable_41Btraining/Adam/Variable_42Btraining/Adam/Variable_43Btraining/Adam/Variable_44Btraining/Adam/Variable_45Btraining/Adam/Variable_46Btraining/Adam/Variable_47Btraining/Adam/Variable_48Btraining/Adam/Variable_49Btraining/Adam/Variable_5Btraining/Adam/Variable_50Btraining/Adam/Variable_51Btraining/Adam/Variable_52Btraining/Adam/Variable_53Btraining/Adam/Variable_54Btraining/Adam/Variable_55Btraining/Adam/Variable_56Btraining/Adam/Variable_57Btraining/Adam/Variable_58Btraining/Adam/Variable_59Btraining/Adam/Variable_6Btraining/Adam/Variable_60Btraining/Adam/Variable_61Btraining/Adam/Variable_62Btraining/Adam/Variable_63Btraining/Adam/Variable_64Btraining/Adam/Variable_65Btraining/Adam/Variable_66Btraining/Adam/Variable_67Btraining/Adam/Variable_68Btraining/Adam/Variable_69Btraining/Adam/Variable_7Btraining/Adam/Variable_70Btraining/Adam/Variable_71Btraining/Adam/Variable_8Btraining/Adam/Variable_9Bup_level_0_no_0/biasBup_level_0_no_0/kernelBup_level_0_no_2/biasBup_level_0_no_2/kernelBup_level_1_no_0/biasBup_level_1_no_0/kernelBup_level_1_no_2/biasBup_level_1_no_2/kernel*
dtype0*
_output_shapes
:e
¬
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*я
value’B“eB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:e
Ц
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*™
_output_shapesЧ
Ф:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*s
dtypesi
g2e	
Ь
save/AssignAssignAdam/beta_1save/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@Adam/beta_1
†
save/Assign_1AssignAdam/beta_2save/RestoreV2:1*
T0*
_class
loc:@Adam/beta_2*
validate_shape(*
_output_shapes
: *
use_locking(
Ю
save/Assign_2Assign
Adam/decaysave/RestoreV2:2*
T0*
_class
loc:@Adam/decay*
validate_shape(*
_output_shapes
: *
use_locking(
®
save/Assign_3AssignAdam/iterationssave/RestoreV2:3*
T0	*"
_class
loc:@Adam/iterations*
validate_shape(*
_output_shapes
: *
use_locking(
Ш
save/Assign_4AssignAdam/lrsave/RestoreV2:4*
T0*
_class
loc:@Adam/lr*
validate_shape(*
_output_shapes
: *
use_locking(
®
save/Assign_5Assignconv2d_1/biassave/RestoreV2:5*
T0* 
_class
loc:@conv2d_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Є
save/Assign_6Assignconv2d_1/kernelsave/RestoreV2:6*
use_locking(*
T0*"
_class
loc:@conv2d_1/kernel*
validate_shape(*&
_output_shapes
: 
®
save/Assign_7Assignconv2d_2/biassave/RestoreV2:7*
T0* 
_class
loc:@conv2d_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Є
save/Assign_8Assignconv2d_2/kernelsave/RestoreV2:8*
T0*"
_class
loc:@conv2d_2/kernel*
validate_shape(*&
_output_shapes
: *
use_locking(
Ї
save/Assign_9Assigndown_level_0_no_0/biassave/RestoreV2:9*
use_locking(*
T0*)
_class
loc:@down_level_0_no_0/bias*
validate_shape(*
_output_shapes
: 
ћ
save/Assign_10Assigndown_level_0_no_0/kernelsave/RestoreV2:10*
use_locking(*
T0*+
_class!
loc:@down_level_0_no_0/kernel*
validate_shape(*&
_output_shapes
: 
Љ
save/Assign_11Assigndown_level_0_no_1/biassave/RestoreV2:11*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*)
_class
loc:@down_level_0_no_1/bias
ћ
save/Assign_12Assigndown_level_0_no_1/kernelsave/RestoreV2:12*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*+
_class!
loc:@down_level_0_no_1/kernel
Љ
save/Assign_13Assigndown_level_1_no_0/biassave/RestoreV2:13*
use_locking(*
T0*)
_class
loc:@down_level_1_no_0/bias*
validate_shape(*
_output_shapes
:@
ћ
save/Assign_14Assigndown_level_1_no_0/kernelsave/RestoreV2:14*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*+
_class!
loc:@down_level_1_no_0/kernel
Љ
save/Assign_15Assigndown_level_1_no_1/biassave/RestoreV2:15*
use_locking(*
T0*)
_class
loc:@down_level_1_no_1/bias*
validate_shape(*
_output_shapes
:@
ћ
save/Assign_16Assigndown_level_1_no_1/kernelsave/RestoreV2:16*
T0*+
_class!
loc:@down_level_1_no_1/kernel*
validate_shape(*&
_output_shapes
:@@*
use_locking(
Ђ
save/Assign_17Assignmiddle_0/biassave/RestoreV2:17*
use_locking(*
T0* 
_class
loc:@middle_0/bias*
validate_shape(*
_output_shapes	
:А
ї
save/Assign_18Assignmiddle_0/kernelsave/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@middle_0/kernel*
validate_shape(*'
_output_shapes
:@А
™
save/Assign_19Assignmiddle_2/biassave/RestoreV2:19*
use_locking(*
T0* 
_class
loc:@middle_2/bias*
validate_shape(*
_output_shapes
:@
ї
save/Assign_20Assignmiddle_2/kernelsave/RestoreV2:20*
validate_shape(*'
_output_shapes
:А@*
use_locking(*
T0*"
_class
loc:@middle_2/kernel
»
save/Assign_21Assigntraining/Adam/Variablesave/RestoreV2:21*
use_locking(*
T0*)
_class
loc:@training/Adam/Variable*
validate_shape(*&
_output_shapes
: 
ј
save/Assign_22Assigntraining/Adam/Variable_1save/RestoreV2:22*
T0*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
_output_shapes
: *
use_locking(
ѕ
save/Assign_23Assigntraining/Adam/Variable_10save/RestoreV2:23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*'
_output_shapes
:А@
¬
save/Assign_24Assigntraining/Adam/Variable_11save/RestoreV2:24*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
validate_shape(*
_output_shapes
:@
ѕ
save/Assign_25Assigntraining/Adam/Variable_12save/RestoreV2:25*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*'
_output_shapes
:А@
¬
save/Assign_26Assigntraining/Adam/Variable_13save/RestoreV2:26*
T0*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
_output_shapes
:@*
use_locking(
ќ
save/Assign_27Assigntraining/Adam/Variable_14save/RestoreV2:27*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_14*
validate_shape(*&
_output_shapes
:@ 
¬
save/Assign_28Assigntraining/Adam/Variable_15save/RestoreV2:28*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15
ќ
save/Assign_29Assigntraining/Adam/Variable_16save/RestoreV2:29*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_16*
validate_shape(*&
_output_shapes
:@ 
¬
save/Assign_30Assigntraining/Adam/Variable_17save/RestoreV2:30*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_17*
validate_shape(*
_output_shapes
: 
ќ
save/Assign_31Assigntraining/Adam/Variable_18save/RestoreV2:31*
T0*,
_class"
 loc:@training/Adam/Variable_18*
validate_shape(*&
_output_shapes
:  *
use_locking(
¬
save/Assign_32Assigntraining/Adam/Variable_19save/RestoreV2:32*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_19
ћ
save/Assign_33Assigntraining/Adam/Variable_2save/RestoreV2:33*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*&
_output_shapes
:  *
use_locking(
ќ
save/Assign_34Assigntraining/Adam/Variable_20save/RestoreV2:34*
T0*,
_class"
 loc:@training/Adam/Variable_20*
validate_shape(*&
_output_shapes
: *
use_locking(
¬
save/Assign_35Assigntraining/Adam/Variable_21save/RestoreV2:35*
T0*,
_class"
 loc:@training/Adam/Variable_21*
validate_shape(*
_output_shapes
:*
use_locking(
ќ
save/Assign_36Assigntraining/Adam/Variable_22save/RestoreV2:36*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_22*
validate_shape(*&
_output_shapes
: 
¬
save/Assign_37Assigntraining/Adam/Variable_23save/RestoreV2:37*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_23
ќ
save/Assign_38Assigntraining/Adam/Variable_24save/RestoreV2:38*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_24*
validate_shape(*&
_output_shapes
: 
¬
save/Assign_39Assigntraining/Adam/Variable_25save/RestoreV2:39*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_25*
validate_shape(*
_output_shapes
: 
ќ
save/Assign_40Assigntraining/Adam/Variable_26save/RestoreV2:40*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_26
¬
save/Assign_41Assigntraining/Adam/Variable_27save/RestoreV2:41*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_27*
validate_shape(*
_output_shapes
: 
ќ
save/Assign_42Assigntraining/Adam/Variable_28save/RestoreV2:42*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_28*
validate_shape(*&
_output_shapes
: @
¬
save/Assign_43Assigntraining/Adam/Variable_29save/RestoreV2:43*
T0*,
_class"
 loc:@training/Adam/Variable_29*
validate_shape(*
_output_shapes
:@*
use_locking(
ј
save/Assign_44Assigntraining/Adam/Variable_3save/RestoreV2:44*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes
: 
ќ
save/Assign_45Assigntraining/Adam/Variable_30save/RestoreV2:45*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_30
¬
save/Assign_46Assigntraining/Adam/Variable_31save/RestoreV2:46*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_31*
validate_shape(*
_output_shapes
:@
ѕ
save/Assign_47Assigntraining/Adam/Variable_32save/RestoreV2:47*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_32*
validate_shape(*'
_output_shapes
:@А
√
save/Assign_48Assigntraining/Adam/Variable_33save/RestoreV2:48*
T0*,
_class"
 loc:@training/Adam/Variable_33*
validate_shape(*
_output_shapes	
:А*
use_locking(
ѕ
save/Assign_49Assigntraining/Adam/Variable_34save/RestoreV2:49*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_34*
validate_shape(*'
_output_shapes
:А@
¬
save/Assign_50Assigntraining/Adam/Variable_35save/RestoreV2:50*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_35*
validate_shape(*
_output_shapes
:@
ѕ
save/Assign_51Assigntraining/Adam/Variable_36save/RestoreV2:51*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_36*
validate_shape(*'
_output_shapes
:А@
¬
save/Assign_52Assigntraining/Adam/Variable_37save/RestoreV2:52*
T0*,
_class"
 loc:@training/Adam/Variable_37*
validate_shape(*
_output_shapes
:@*
use_locking(
ќ
save/Assign_53Assigntraining/Adam/Variable_38save/RestoreV2:53*
validate_shape(*&
_output_shapes
:@ *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_38
¬
save/Assign_54Assigntraining/Adam/Variable_39save/RestoreV2:54*
T0*,
_class"
 loc:@training/Adam/Variable_39*
validate_shape(*
_output_shapes
: *
use_locking(
ћ
save/Assign_55Assigntraining/Adam/Variable_4save/RestoreV2:55*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_4
ќ
save/Assign_56Assigntraining/Adam/Variable_40save/RestoreV2:56*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_40*
validate_shape(*&
_output_shapes
:@ 
¬
save/Assign_57Assigntraining/Adam/Variable_41save/RestoreV2:57*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_41
ќ
save/Assign_58Assigntraining/Adam/Variable_42save/RestoreV2:58*
T0*,
_class"
 loc:@training/Adam/Variable_42*
validate_shape(*&
_output_shapes
:  *
use_locking(
¬
save/Assign_59Assigntraining/Adam/Variable_43save/RestoreV2:59*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_43
ќ
save/Assign_60Assigntraining/Adam/Variable_44save/RestoreV2:60*
T0*,
_class"
 loc:@training/Adam/Variable_44*
validate_shape(*&
_output_shapes
: *
use_locking(
¬
save/Assign_61Assigntraining/Adam/Variable_45save/RestoreV2:61*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_45*
validate_shape(*
_output_shapes
:
ќ
save/Assign_62Assigntraining/Adam/Variable_46save/RestoreV2:62*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_46*
validate_shape(*&
_output_shapes
: 
¬
save/Assign_63Assigntraining/Adam/Variable_47save/RestoreV2:63*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_47*
validate_shape(*
_output_shapes
:
¬
save/Assign_64Assigntraining/Adam/Variable_48save/RestoreV2:64*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_48
¬
save/Assign_65Assigntraining/Adam/Variable_49save/RestoreV2:65*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_49*
validate_shape(*
_output_shapes
:
ј
save/Assign_66Assigntraining/Adam/Variable_5save/RestoreV2:66*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
_output_shapes
:@
¬
save/Assign_67Assigntraining/Adam/Variable_50save/RestoreV2:67*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_50
¬
save/Assign_68Assigntraining/Adam/Variable_51save/RestoreV2:68*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_51*
validate_shape(*
_output_shapes
:
¬
save/Assign_69Assigntraining/Adam/Variable_52save/RestoreV2:69*
T0*,
_class"
 loc:@training/Adam/Variable_52*
validate_shape(*
_output_shapes
:*
use_locking(
¬
save/Assign_70Assigntraining/Adam/Variable_53save/RestoreV2:70*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_53*
validate_shape(*
_output_shapes
:
¬
save/Assign_71Assigntraining/Adam/Variable_54save/RestoreV2:71*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_54*
validate_shape(*
_output_shapes
:
¬
save/Assign_72Assigntraining/Adam/Variable_55save/RestoreV2:72*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_55*
validate_shape(*
_output_shapes
:
¬
save/Assign_73Assigntraining/Adam/Variable_56save/RestoreV2:73*
T0*,
_class"
 loc:@training/Adam/Variable_56*
validate_shape(*
_output_shapes
:*
use_locking(
¬
save/Assign_74Assigntraining/Adam/Variable_57save/RestoreV2:74*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_57*
validate_shape(*
_output_shapes
:
¬
save/Assign_75Assigntraining/Adam/Variable_58save/RestoreV2:75*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_58*
validate_shape(*
_output_shapes
:
¬
save/Assign_76Assigntraining/Adam/Variable_59save/RestoreV2:76*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_59
ћ
save/Assign_77Assigntraining/Adam/Variable_6save/RestoreV2:77*
T0*+
_class!
loc:@training/Adam/Variable_6*
validate_shape(*&
_output_shapes
:@@*
use_locking(
¬
save/Assign_78Assigntraining/Adam/Variable_60save/RestoreV2:78*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_60*
validate_shape(*
_output_shapes
:
¬
save/Assign_79Assigntraining/Adam/Variable_61save/RestoreV2:79*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_61*
validate_shape(*
_output_shapes
:
¬
save/Assign_80Assigntraining/Adam/Variable_62save/RestoreV2:80*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_62
¬
save/Assign_81Assigntraining/Adam/Variable_63save/RestoreV2:81*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_63*
validate_shape(*
_output_shapes
:
¬
save/Assign_82Assigntraining/Adam/Variable_64save/RestoreV2:82*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_64*
validate_shape(*
_output_shapes
:
¬
save/Assign_83Assigntraining/Adam/Variable_65save/RestoreV2:83*
T0*,
_class"
 loc:@training/Adam/Variable_65*
validate_shape(*
_output_shapes
:*
use_locking(
¬
save/Assign_84Assigntraining/Adam/Variable_66save/RestoreV2:84*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_66*
validate_shape(*
_output_shapes
:
¬
save/Assign_85Assigntraining/Adam/Variable_67save/RestoreV2:85*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_67*
validate_shape(*
_output_shapes
:
¬
save/Assign_86Assigntraining/Adam/Variable_68save/RestoreV2:86*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_68*
validate_shape(*
_output_shapes
:
¬
save/Assign_87Assigntraining/Adam/Variable_69save/RestoreV2:87*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_69*
validate_shape(*
_output_shapes
:
ј
save/Assign_88Assigntraining/Adam/Variable_7save/RestoreV2:88*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
:@
¬
save/Assign_89Assigntraining/Adam/Variable_70save/RestoreV2:89*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_70*
validate_shape(*
_output_shapes
:
¬
save/Assign_90Assigntraining/Adam/Variable_71save/RestoreV2:90*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_71
Ќ
save/Assign_91Assigntraining/Adam/Variable_8save/RestoreV2:91*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*'
_output_shapes
:@А
Ѕ
save/Assign_92Assigntraining/Adam/Variable_9save/RestoreV2:92*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9
Є
save/Assign_93Assignup_level_0_no_0/biassave/RestoreV2:93*
use_locking(*
T0*'
_class
loc:@up_level_0_no_0/bias*
validate_shape(*
_output_shapes
: 
»
save/Assign_94Assignup_level_0_no_0/kernelsave/RestoreV2:94*
use_locking(*
T0*)
_class
loc:@up_level_0_no_0/kernel*
validate_shape(*&
_output_shapes
:@ 
Є
save/Assign_95Assignup_level_0_no_2/biassave/RestoreV2:95*
T0*'
_class
loc:@up_level_0_no_2/bias*
validate_shape(*
_output_shapes
: *
use_locking(
»
save/Assign_96Assignup_level_0_no_2/kernelsave/RestoreV2:96*
T0*)
_class
loc:@up_level_0_no_2/kernel*
validate_shape(*&
_output_shapes
:  *
use_locking(
Є
save/Assign_97Assignup_level_1_no_0/biassave/RestoreV2:97*
use_locking(*
T0*'
_class
loc:@up_level_1_no_0/bias*
validate_shape(*
_output_shapes
:@
…
save/Assign_98Assignup_level_1_no_0/kernelsave/RestoreV2:98*
validate_shape(*'
_output_shapes
:А@*
use_locking(*
T0*)
_class
loc:@up_level_1_no_0/kernel
Є
save/Assign_99Assignup_level_1_no_2/biassave/RestoreV2:99*
T0*'
_class
loc:@up_level_1_no_2/bias*
validate_shape(*
_output_shapes
: *
use_locking(
 
save/Assign_100Assignup_level_1_no_2/kernelsave/RestoreV2:100*
use_locking(*
T0*)
_class
loc:@up_level_1_no_2/kernel*
validate_shape(*&
_output_shapes
:@ 
ƒ
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"„_
trainable_variablesњ_Љ_
Д
down_level_0_no_0/kernel:0down_level_0_no_0/kernel/Assigndown_level_0_no_0/kernel/read:02"down_level_0_no_0/random_uniform:08
u
down_level_0_no_0/bias:0down_level_0_no_0/bias/Assigndown_level_0_no_0/bias/read:02down_level_0_no_0/Const:08
Д
down_level_0_no_1/kernel:0down_level_0_no_1/kernel/Assigndown_level_0_no_1/kernel/read:02"down_level_0_no_1/random_uniform:08
u
down_level_0_no_1/bias:0down_level_0_no_1/bias/Assigndown_level_0_no_1/bias/read:02down_level_0_no_1/Const:08
Д
down_level_1_no_0/kernel:0down_level_1_no_0/kernel/Assigndown_level_1_no_0/kernel/read:02"down_level_1_no_0/random_uniform:08
u
down_level_1_no_0/bias:0down_level_1_no_0/bias/Assigndown_level_1_no_0/bias/read:02down_level_1_no_0/Const:08
Д
down_level_1_no_1/kernel:0down_level_1_no_1/kernel/Assigndown_level_1_no_1/kernel/read:02"down_level_1_no_1/random_uniform:08
u
down_level_1_no_1/bias:0down_level_1_no_1/bias/Assigndown_level_1_no_1/bias/read:02down_level_1_no_1/Const:08
`
middle_0/kernel:0middle_0/kernel/Assignmiddle_0/kernel/read:02middle_0/random_uniform:08
Q
middle_0/bias:0middle_0/bias/Assignmiddle_0/bias/read:02middle_0/Const:08
`
middle_2/kernel:0middle_2/kernel/Assignmiddle_2/kernel/read:02middle_2/random_uniform:08
Q
middle_2/bias:0middle_2/bias/Assignmiddle_2/bias/read:02middle_2/Const:08
|
up_level_1_no_0/kernel:0up_level_1_no_0/kernel/Assignup_level_1_no_0/kernel/read:02 up_level_1_no_0/random_uniform:08
m
up_level_1_no_0/bias:0up_level_1_no_0/bias/Assignup_level_1_no_0/bias/read:02up_level_1_no_0/Const:08
|
up_level_1_no_2/kernel:0up_level_1_no_2/kernel/Assignup_level_1_no_2/kernel/read:02 up_level_1_no_2/random_uniform:08
m
up_level_1_no_2/bias:0up_level_1_no_2/bias/Assignup_level_1_no_2/bias/read:02up_level_1_no_2/Const:08
|
up_level_0_no_0/kernel:0up_level_0_no_0/kernel/Assignup_level_0_no_0/kernel/read:02 up_level_0_no_0/random_uniform:08
m
up_level_0_no_0/bias:0up_level_0_no_0/bias/Assignup_level_0_no_0/bias/read:02up_level_0_no_0/Const:08
|
up_level_0_no_2/kernel:0up_level_0_no_2/kernel/Assignup_level_0_no_2/kernel/read:02 up_level_0_no_2/random_uniform:08
m
up_level_0_no_2/bias:0up_level_0_no_2/bias/Assignup_level_0_no_2/bias/read:02up_level_0_no_2/Const:08
`
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/random_uniform:08
Q
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:08
`
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02conv2d_2/random_uniform:08
Q
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02conv2d_2/Const:08
f
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:08
F
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:08
V
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:08
V
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:08
R
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:08
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/zeros:08
y
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/zeros_1:08
y
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/zeros_2:08
y
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/zeros_3:08
y
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/zeros_4:08
y
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/zeros_5:08
y
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/zeros_6:08
y
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/zeros_7:08
y
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/zeros_8:08
y
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/zeros_9:08
}
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/zeros_10:08
}
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/zeros_11:08
}
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/zeros_12:08
}
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/zeros_13:08
}
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/zeros_14:08
}
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/zeros_15:08
}
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign training/Adam/Variable_16/read:02training/Adam/zeros_16:08
}
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08
}
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign training/Adam/Variable_18/read:02training/Adam/zeros_18:08
}
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign training/Adam/Variable_19/read:02training/Adam/zeros_19:08
}
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign training/Adam/Variable_20/read:02training/Adam/zeros_20:08
}
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign training/Adam/Variable_21/read:02training/Adam/zeros_21:08
}
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign training/Adam/Variable_22/read:02training/Adam/zeros_22:08
}
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08
}
training/Adam/Variable_24:0 training/Adam/Variable_24/Assign training/Adam/Variable_24/read:02training/Adam/zeros_24:08
}
training/Adam/Variable_25:0 training/Adam/Variable_25/Assign training/Adam/Variable_25/read:02training/Adam/zeros_25:08
}
training/Adam/Variable_26:0 training/Adam/Variable_26/Assign training/Adam/Variable_26/read:02training/Adam/zeros_26:08
}
training/Adam/Variable_27:0 training/Adam/Variable_27/Assign training/Adam/Variable_27/read:02training/Adam/zeros_27:08
}
training/Adam/Variable_28:0 training/Adam/Variable_28/Assign training/Adam/Variable_28/read:02training/Adam/zeros_28:08
}
training/Adam/Variable_29:0 training/Adam/Variable_29/Assign training/Adam/Variable_29/read:02training/Adam/zeros_29:08
}
training/Adam/Variable_30:0 training/Adam/Variable_30/Assign training/Adam/Variable_30/read:02training/Adam/zeros_30:08
}
training/Adam/Variable_31:0 training/Adam/Variable_31/Assign training/Adam/Variable_31/read:02training/Adam/zeros_31:08
}
training/Adam/Variable_32:0 training/Adam/Variable_32/Assign training/Adam/Variable_32/read:02training/Adam/zeros_32:08
}
training/Adam/Variable_33:0 training/Adam/Variable_33/Assign training/Adam/Variable_33/read:02training/Adam/zeros_33:08
}
training/Adam/Variable_34:0 training/Adam/Variable_34/Assign training/Adam/Variable_34/read:02training/Adam/zeros_34:08
}
training/Adam/Variable_35:0 training/Adam/Variable_35/Assign training/Adam/Variable_35/read:02training/Adam/zeros_35:08
}
training/Adam/Variable_36:0 training/Adam/Variable_36/Assign training/Adam/Variable_36/read:02training/Adam/zeros_36:08
}
training/Adam/Variable_37:0 training/Adam/Variable_37/Assign training/Adam/Variable_37/read:02training/Adam/zeros_37:08
}
training/Adam/Variable_38:0 training/Adam/Variable_38/Assign training/Adam/Variable_38/read:02training/Adam/zeros_38:08
}
training/Adam/Variable_39:0 training/Adam/Variable_39/Assign training/Adam/Variable_39/read:02training/Adam/zeros_39:08
}
training/Adam/Variable_40:0 training/Adam/Variable_40/Assign training/Adam/Variable_40/read:02training/Adam/zeros_40:08
}
training/Adam/Variable_41:0 training/Adam/Variable_41/Assign training/Adam/Variable_41/read:02training/Adam/zeros_41:08
}
training/Adam/Variable_42:0 training/Adam/Variable_42/Assign training/Adam/Variable_42/read:02training/Adam/zeros_42:08
}
training/Adam/Variable_43:0 training/Adam/Variable_43/Assign training/Adam/Variable_43/read:02training/Adam/zeros_43:08
}
training/Adam/Variable_44:0 training/Adam/Variable_44/Assign training/Adam/Variable_44/read:02training/Adam/zeros_44:08
}
training/Adam/Variable_45:0 training/Adam/Variable_45/Assign training/Adam/Variable_45/read:02training/Adam/zeros_45:08
}
training/Adam/Variable_46:0 training/Adam/Variable_46/Assign training/Adam/Variable_46/read:02training/Adam/zeros_46:08
}
training/Adam/Variable_47:0 training/Adam/Variable_47/Assign training/Adam/Variable_47/read:02training/Adam/zeros_47:08
}
training/Adam/Variable_48:0 training/Adam/Variable_48/Assign training/Adam/Variable_48/read:02training/Adam/zeros_48:08
}
training/Adam/Variable_49:0 training/Adam/Variable_49/Assign training/Adam/Variable_49/read:02training/Adam/zeros_49:08
}
training/Adam/Variable_50:0 training/Adam/Variable_50/Assign training/Adam/Variable_50/read:02training/Adam/zeros_50:08
}
training/Adam/Variable_51:0 training/Adam/Variable_51/Assign training/Adam/Variable_51/read:02training/Adam/zeros_51:08
}
training/Adam/Variable_52:0 training/Adam/Variable_52/Assign training/Adam/Variable_52/read:02training/Adam/zeros_52:08
}
training/Adam/Variable_53:0 training/Adam/Variable_53/Assign training/Adam/Variable_53/read:02training/Adam/zeros_53:08
}
training/Adam/Variable_54:0 training/Adam/Variable_54/Assign training/Adam/Variable_54/read:02training/Adam/zeros_54:08
}
training/Adam/Variable_55:0 training/Adam/Variable_55/Assign training/Adam/Variable_55/read:02training/Adam/zeros_55:08
}
training/Adam/Variable_56:0 training/Adam/Variable_56/Assign training/Adam/Variable_56/read:02training/Adam/zeros_56:08
}
training/Adam/Variable_57:0 training/Adam/Variable_57/Assign training/Adam/Variable_57/read:02training/Adam/zeros_57:08
}
training/Adam/Variable_58:0 training/Adam/Variable_58/Assign training/Adam/Variable_58/read:02training/Adam/zeros_58:08
}
training/Adam/Variable_59:0 training/Adam/Variable_59/Assign training/Adam/Variable_59/read:02training/Adam/zeros_59:08
}
training/Adam/Variable_60:0 training/Adam/Variable_60/Assign training/Adam/Variable_60/read:02training/Adam/zeros_60:08
}
training/Adam/Variable_61:0 training/Adam/Variable_61/Assign training/Adam/Variable_61/read:02training/Adam/zeros_61:08
}
training/Adam/Variable_62:0 training/Adam/Variable_62/Assign training/Adam/Variable_62/read:02training/Adam/zeros_62:08
}
training/Adam/Variable_63:0 training/Adam/Variable_63/Assign training/Adam/Variable_63/read:02training/Adam/zeros_63:08
}
training/Adam/Variable_64:0 training/Adam/Variable_64/Assign training/Adam/Variable_64/read:02training/Adam/zeros_64:08
}
training/Adam/Variable_65:0 training/Adam/Variable_65/Assign training/Adam/Variable_65/read:02training/Adam/zeros_65:08
}
training/Adam/Variable_66:0 training/Adam/Variable_66/Assign training/Adam/Variable_66/read:02training/Adam/zeros_66:08
}
training/Adam/Variable_67:0 training/Adam/Variable_67/Assign training/Adam/Variable_67/read:02training/Adam/zeros_67:08
}
training/Adam/Variable_68:0 training/Adam/Variable_68/Assign training/Adam/Variable_68/read:02training/Adam/zeros_68:08
}
training/Adam/Variable_69:0 training/Adam/Variable_69/Assign training/Adam/Variable_69/read:02training/Adam/zeros_69:08
}
training/Adam/Variable_70:0 training/Adam/Variable_70/Assign training/Adam/Variable_70/read:02training/Adam/zeros_70:08
}
training/Adam/Variable_71:0 training/Adam/Variable_71/Assign training/Adam/Variable_71/read:02training/Adam/zeros_71:08"Q
	summariesD
B
net_input:0
net_target:0
net_output_mean:0
net_output_scale:0"Ќ_
	variablesњ_Љ_
Д
down_level_0_no_0/kernel:0down_level_0_no_0/kernel/Assigndown_level_0_no_0/kernel/read:02"down_level_0_no_0/random_uniform:08
u
down_level_0_no_0/bias:0down_level_0_no_0/bias/Assigndown_level_0_no_0/bias/read:02down_level_0_no_0/Const:08
Д
down_level_0_no_1/kernel:0down_level_0_no_1/kernel/Assigndown_level_0_no_1/kernel/read:02"down_level_0_no_1/random_uniform:08
u
down_level_0_no_1/bias:0down_level_0_no_1/bias/Assigndown_level_0_no_1/bias/read:02down_level_0_no_1/Const:08
Д
down_level_1_no_0/kernel:0down_level_1_no_0/kernel/Assigndown_level_1_no_0/kernel/read:02"down_level_1_no_0/random_uniform:08
u
down_level_1_no_0/bias:0down_level_1_no_0/bias/Assigndown_level_1_no_0/bias/read:02down_level_1_no_0/Const:08
Д
down_level_1_no_1/kernel:0down_level_1_no_1/kernel/Assigndown_level_1_no_1/kernel/read:02"down_level_1_no_1/random_uniform:08
u
down_level_1_no_1/bias:0down_level_1_no_1/bias/Assigndown_level_1_no_1/bias/read:02down_level_1_no_1/Const:08
`
middle_0/kernel:0middle_0/kernel/Assignmiddle_0/kernel/read:02middle_0/random_uniform:08
Q
middle_0/bias:0middle_0/bias/Assignmiddle_0/bias/read:02middle_0/Const:08
`
middle_2/kernel:0middle_2/kernel/Assignmiddle_2/kernel/read:02middle_2/random_uniform:08
Q
middle_2/bias:0middle_2/bias/Assignmiddle_2/bias/read:02middle_2/Const:08
|
up_level_1_no_0/kernel:0up_level_1_no_0/kernel/Assignup_level_1_no_0/kernel/read:02 up_level_1_no_0/random_uniform:08
m
up_level_1_no_0/bias:0up_level_1_no_0/bias/Assignup_level_1_no_0/bias/read:02up_level_1_no_0/Const:08
|
up_level_1_no_2/kernel:0up_level_1_no_2/kernel/Assignup_level_1_no_2/kernel/read:02 up_level_1_no_2/random_uniform:08
m
up_level_1_no_2/bias:0up_level_1_no_2/bias/Assignup_level_1_no_2/bias/read:02up_level_1_no_2/Const:08
|
up_level_0_no_0/kernel:0up_level_0_no_0/kernel/Assignup_level_0_no_0/kernel/read:02 up_level_0_no_0/random_uniform:08
m
up_level_0_no_0/bias:0up_level_0_no_0/bias/Assignup_level_0_no_0/bias/read:02up_level_0_no_0/Const:08
|
up_level_0_no_2/kernel:0up_level_0_no_2/kernel/Assignup_level_0_no_2/kernel/read:02 up_level_0_no_2/random_uniform:08
m
up_level_0_no_2/bias:0up_level_0_no_2/bias/Assignup_level_0_no_2/bias/read:02up_level_0_no_2/Const:08
`
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02conv2d_1/random_uniform:08
Q
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02conv2d_1/Const:08
`
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02conv2d_2/random_uniform:08
Q
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02conv2d_2/Const:08
f
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:08
F
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:08
V
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:08
V
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:08
R
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:08
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/zeros:08
y
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/zeros_1:08
y
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/zeros_2:08
y
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/zeros_3:08
y
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/zeros_4:08
y
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/zeros_5:08
y
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/zeros_6:08
y
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/zeros_7:08
y
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/zeros_8:08
y
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/zeros_9:08
}
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/zeros_10:08
}
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/zeros_11:08
}
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/zeros_12:08
}
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/zeros_13:08
}
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/zeros_14:08
}
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/zeros_15:08
}
training/Adam/Variable_16:0 training/Adam/Variable_16/Assign training/Adam/Variable_16/read:02training/Adam/zeros_16:08
}
training/Adam/Variable_17:0 training/Adam/Variable_17/Assign training/Adam/Variable_17/read:02training/Adam/zeros_17:08
}
training/Adam/Variable_18:0 training/Adam/Variable_18/Assign training/Adam/Variable_18/read:02training/Adam/zeros_18:08
}
training/Adam/Variable_19:0 training/Adam/Variable_19/Assign training/Adam/Variable_19/read:02training/Adam/zeros_19:08
}
training/Adam/Variable_20:0 training/Adam/Variable_20/Assign training/Adam/Variable_20/read:02training/Adam/zeros_20:08
}
training/Adam/Variable_21:0 training/Adam/Variable_21/Assign training/Adam/Variable_21/read:02training/Adam/zeros_21:08
}
training/Adam/Variable_22:0 training/Adam/Variable_22/Assign training/Adam/Variable_22/read:02training/Adam/zeros_22:08
}
training/Adam/Variable_23:0 training/Adam/Variable_23/Assign training/Adam/Variable_23/read:02training/Adam/zeros_23:08
}
training/Adam/Variable_24:0 training/Adam/Variable_24/Assign training/Adam/Variable_24/read:02training/Adam/zeros_24:08
}
training/Adam/Variable_25:0 training/Adam/Variable_25/Assign training/Adam/Variable_25/read:02training/Adam/zeros_25:08
}
training/Adam/Variable_26:0 training/Adam/Variable_26/Assign training/Adam/Variable_26/read:02training/Adam/zeros_26:08
}
training/Adam/Variable_27:0 training/Adam/Variable_27/Assign training/Adam/Variable_27/read:02training/Adam/zeros_27:08
}
training/Adam/Variable_28:0 training/Adam/Variable_28/Assign training/Adam/Variable_28/read:02training/Adam/zeros_28:08
}
training/Adam/Variable_29:0 training/Adam/Variable_29/Assign training/Adam/Variable_29/read:02training/Adam/zeros_29:08
}
training/Adam/Variable_30:0 training/Adam/Variable_30/Assign training/Adam/Variable_30/read:02training/Adam/zeros_30:08
}
training/Adam/Variable_31:0 training/Adam/Variable_31/Assign training/Adam/Variable_31/read:02training/Adam/zeros_31:08
}
training/Adam/Variable_32:0 training/Adam/Variable_32/Assign training/Adam/Variable_32/read:02training/Adam/zeros_32:08
}
training/Adam/Variable_33:0 training/Adam/Variable_33/Assign training/Adam/Variable_33/read:02training/Adam/zeros_33:08
}
training/Adam/Variable_34:0 training/Adam/Variable_34/Assign training/Adam/Variable_34/read:02training/Adam/zeros_34:08
}
training/Adam/Variable_35:0 training/Adam/Variable_35/Assign training/Adam/Variable_35/read:02training/Adam/zeros_35:08
}
training/Adam/Variable_36:0 training/Adam/Variable_36/Assign training/Adam/Variable_36/read:02training/Adam/zeros_36:08
}
training/Adam/Variable_37:0 training/Adam/Variable_37/Assign training/Adam/Variable_37/read:02training/Adam/zeros_37:08
}
training/Adam/Variable_38:0 training/Adam/Variable_38/Assign training/Adam/Variable_38/read:02training/Adam/zeros_38:08
}
training/Adam/Variable_39:0 training/Adam/Variable_39/Assign training/Adam/Variable_39/read:02training/Adam/zeros_39:08
}
training/Adam/Variable_40:0 training/Adam/Variable_40/Assign training/Adam/Variable_40/read:02training/Adam/zeros_40:08
}
training/Adam/Variable_41:0 training/Adam/Variable_41/Assign training/Adam/Variable_41/read:02training/Adam/zeros_41:08
}
training/Adam/Variable_42:0 training/Adam/Variable_42/Assign training/Adam/Variable_42/read:02training/Adam/zeros_42:08
}
training/Adam/Variable_43:0 training/Adam/Variable_43/Assign training/Adam/Variable_43/read:02training/Adam/zeros_43:08
}
training/Adam/Variable_44:0 training/Adam/Variable_44/Assign training/Adam/Variable_44/read:02training/Adam/zeros_44:08
}
training/Adam/Variable_45:0 training/Adam/Variable_45/Assign training/Adam/Variable_45/read:02training/Adam/zeros_45:08
}
training/Adam/Variable_46:0 training/Adam/Variable_46/Assign training/Adam/Variable_46/read:02training/Adam/zeros_46:08
}
training/Adam/Variable_47:0 training/Adam/Variable_47/Assign training/Adam/Variable_47/read:02training/Adam/zeros_47:08
}
training/Adam/Variable_48:0 training/Adam/Variable_48/Assign training/Adam/Variable_48/read:02training/Adam/zeros_48:08
}
training/Adam/Variable_49:0 training/Adam/Variable_49/Assign training/Adam/Variable_49/read:02training/Adam/zeros_49:08
}
training/Adam/Variable_50:0 training/Adam/Variable_50/Assign training/Adam/Variable_50/read:02training/Adam/zeros_50:08
}
training/Adam/Variable_51:0 training/Adam/Variable_51/Assign training/Adam/Variable_51/read:02training/Adam/zeros_51:08
}
training/Adam/Variable_52:0 training/Adam/Variable_52/Assign training/Adam/Variable_52/read:02training/Adam/zeros_52:08
}
training/Adam/Variable_53:0 training/Adam/Variable_53/Assign training/Adam/Variable_53/read:02training/Adam/zeros_53:08
}
training/Adam/Variable_54:0 training/Adam/Variable_54/Assign training/Adam/Variable_54/read:02training/Adam/zeros_54:08
}
training/Adam/Variable_55:0 training/Adam/Variable_55/Assign training/Adam/Variable_55/read:02training/Adam/zeros_55:08
}
training/Adam/Variable_56:0 training/Adam/Variable_56/Assign training/Adam/Variable_56/read:02training/Adam/zeros_56:08
}
training/Adam/Variable_57:0 training/Adam/Variable_57/Assign training/Adam/Variable_57/read:02training/Adam/zeros_57:08
}
training/Adam/Variable_58:0 training/Adam/Variable_58/Assign training/Adam/Variable_58/read:02training/Adam/zeros_58:08
}
training/Adam/Variable_59:0 training/Adam/Variable_59/Assign training/Adam/Variable_59/read:02training/Adam/zeros_59:08
}
training/Adam/Variable_60:0 training/Adam/Variable_60/Assign training/Adam/Variable_60/read:02training/Adam/zeros_60:08
}
training/Adam/Variable_61:0 training/Adam/Variable_61/Assign training/Adam/Variable_61/read:02training/Adam/zeros_61:08
}
training/Adam/Variable_62:0 training/Adam/Variable_62/Assign training/Adam/Variable_62/read:02training/Adam/zeros_62:08
}
training/Adam/Variable_63:0 training/Adam/Variable_63/Assign training/Adam/Variable_63/read:02training/Adam/zeros_63:08
}
training/Adam/Variable_64:0 training/Adam/Variable_64/Assign training/Adam/Variable_64/read:02training/Adam/zeros_64:08
}
training/Adam/Variable_65:0 training/Adam/Variable_65/Assign training/Adam/Variable_65/read:02training/Adam/zeros_65:08
}
training/Adam/Variable_66:0 training/Adam/Variable_66/Assign training/Adam/Variable_66/read:02training/Adam/zeros_66:08
}
training/Adam/Variable_67:0 training/Adam/Variable_67/Assign training/Adam/Variable_67/read:02training/Adam/zeros_67:08
}
training/Adam/Variable_68:0 training/Adam/Variable_68/Assign training/Adam/Variable_68/read:02training/Adam/zeros_68:08
}
training/Adam/Variable_69:0 training/Adam/Variable_69/Assign training/Adam/Variable_69/read:02training/Adam/zeros_69:08
}
training/Adam/Variable_70:0 training/Adam/Variable_70/Assign training/Adam/Variable_70/read:02training/Adam/zeros_70:08
}
training/Adam/Variable_71:0 training/Adam/Variable_71/Assign training/Adam/Variable_71/read:02training/Adam/zeros_71:08*∆
serving_default≤
A
input8
input:0+€€€€€€€€€€€€€€€€€€€€€€€€€€€Q
outputG
concatenate_3/concat:0+€€€€€€€€€€€€€€€€€€€€€€€€€€€tensorflow/serving/predict