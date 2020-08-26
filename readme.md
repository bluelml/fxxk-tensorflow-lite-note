# 1. Linux 环境和重装

## 1.1 系统安装

popos：custom 安装/ /boot swap 不用安装/home 

## 1.2 启动表修复

grub-repair 修复启动表 进入linux

gurb customizer 修复win启动表

## 1.3 挂载home盘

自动挂载原home盘 runtime覆盖原有根目录下自带home盘

```shell
vim /etc/fstab 
# 添加如下行
/dev/sda5  /home  ext4  defaults  0  0 
```

## 1.4 应用修复 zsh修复

```shell
apt install vim tmux ranger zsh flameshot ntpdate make
chsh -s /bin/zsh #输入密码 重启 修复zsh
```

安装chrome vscode 搜狗输入法 卸载ibus 安装fcitx

## 1.5 统一双系统时间

```shell
sudo ntpdate time.windows.com 
sudo hwclock --localtime --systohc 
```


## 1.6 设置免密码sudo     

```shell
vim /etc/sudoers  
# 更改用户组的nopasswd 除了root全部改称nopasswd格式
%admin ALL=(ALL) NOPASSWD: ALL
```

## 1.7 安装邮箱

mailspring

qq邮箱验证码xwsxzxowqgupbddc当作密码

# 2. GCC

gcc 7.3 

## 2.1 源代码make 或者直接apt install

```shell
wget https://ftp.gnu.org/gnu/gcc/gcc???  # 下载镜像源
sudo make 
sudo make check
sudo make install  #一定要sudo 否则奇怪问题 
```

install 的 bin文件在/usr/local/bin/gcc   并不在/usr/bin（添加源之后apt install 的gcc在这个路径）

## 2.2 多版本并存切换

```sh
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 40 
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50
sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc-7 60
sudo update-alternatives --config gcc  # 控制默认gcc版本 g++同上
```

# 3. Anaconda虚拟环境
## 3.1 安装

sh文件装好 

.zshrc  或者.bashrc注册  source刷新
 注意添加sock5的支持 与本机代理同步：

```shell
unset all_proxy && unset ALL_PROXY
pip install pysocks  #添加sock支持
```

## 3.2 设置虚拟环境

```shell
create -n name python=3.6   #3.6对应了tf1.7版本 
pip install --upgrade pip
```

## 3.3 初次使用添加bazel构建的支持

新建立的conda虚拟环境中 预先安装对应的库

```shell
pip install keras_applications==1.0.4 --no-deps
pip install keras_preprocessing==1.0.2 --no-deps
pip install h5py==2.8.0
```

注意：尽量不要再base环境安装tensorflow和jupyter notebook之类的包，防止虚拟环境没有安装而去上级base环境找到直接使用引起版本错误

# 4. TensorFlow安装和构建

注意：pip方式安装容易引起和本地custom op的补丁版本不一致 建议本地同版本build成whl后安装

对齐GCC和bazel版本

## 4.1 bazel

安装

https://github.com/bazelbuild/bazel/releases  下载对应版本的installer-linux-x86_64.sh

```shell
chmod +x bazelxxxxx.sh 
./bazelxxxxxx.sh --user
```

卸载

```shell
rm -rf ~/.bazel
rm -rf ~/bin
rm -rf /usr/bin/bazel    
```

一般只是切换不需要更改.zshrc文件 先卸载再安装后 用bazel version验证版本

bazel可以用来查看某个目标的所有依赖关系例如：

```shell
bazel query 'deps(//tensorflow/lite/java:tensorflowlite)' --output package      # 查看所有包（包指的是某个有BUILD文件的目录）
bazel query 'deps(//tensorflow/lite/java:tensorflowlite)' --output graph        # 查看依赖图
```

注意：需要先进行各类配置再进行query，例如对于android ndk路径的指定，如果在configure之后制定了build option的ndk路径，在query时仍旧会出错，这是因为query并不是build，这时需要在tensorflow根目录的WORKSPACE文件之中显式指出ndk目录：

```python
android_ndk_repository(
    name = "androidndk",
    path = "/home/gx/android-ndk-r18b",
)
```

对于sdk等问题同样，如果无法query但是能正常build，就需要在工作空间显式写明路径

## 4.2 TensorFlow源码的编译配置configure

### 4.2.1 ./configure先自动配置生成配置文件

第一步询问的python解释器地址 是conda env的解释器 所以需要先启动虚拟环境

clang可以选y    XLA可以选false  cuda为GPU支持  其他都是默认N

workspace 在build安卓aar时使用 本地tensorflow和lite不需要设置 默认N即可

### 4.2.2 vim .tf_configure.bazelrc 再手动修改配置文件

参考GCC手册：对于cpu优化标记  https://gcc.gnu.org/onlinedocs/gcc-4.5.3/gcc/i386-and-x86_002d64-Options.html

更改  build:opt --copt=-march=native（本地支持）   为   build:opt --copt=-mtune=generic（通用支持）

更改  build:opt --host_copt=-march=native   为   build:opt --host_copt=-mtune=generic

注：上述修改的场景是服务器打包并准备在本地使用，服务器cpu和本地cpu不同

## 4.3 build相关

tensorflow目录下的bazel-bin等目录是~/.cache/bazel/...的链接 

如果更换tensorflow版本，需要用bazel clean命令清除残留缓存

### 4.3.1 构建tensorflow

```shell
bazel build //tensorflow/tools/pip_package:build_pip_package --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"   #构建pip打包器 ABI参数为gcc5以上 兼容旧版本 单独环境可以无视

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg   #打为whl包

source activate xxx
pip install /tmp/tensorflow_pkg/tensorflow-cp36xxxx.whl    #安装tensorflow
```

本地机器初次build（i5 cpu 8gb内存）约7小时，需要限制bazel线程数（参见bazel文档），并配好梯子和sock5代理

海外服务器初次build打包速度约2小时左右，原因是bazel动态下载所需的库和文件

修改源文件后，增量build时间很短，具体时间视修改所关联的文件不同，单独文件修改大约十秒级别

### 4.3.2 单独构建tensorflow custom op的.so共享库

tensorflow/core/user_ops目录下

添加新算子的.cc源文件 格式参见tf官网

```shell
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print("".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print("".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared zerof.cc -o zerof.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -D_GLIBCXX_USE_CXX11_ABI=0 -O2
```

这里直接使用了g++构建 zerof.so文件 原因是本地使用bazel会出现google protobuf依赖找不到等诡异问题

```python
tf.load_op_library('??/zerof.so').zerof()  #调用so中的op
```

如果模型中存在tensorflow都没有的算子，有两个办法处理，一是类似这种方式先给tf增加算子，然后在tflite中实现；二是可以直接更改tflite模型的flatbuffer文件（从json中改比较人性化）然后直接tflite中实现即可

### 4.3.3 单独构建python tensorflow lite解释器的.so共享库

注意：

​    之前版本使用SWIG进行cpp和python之间的交互   后续tf官方将改为pybind11的方式实现交互

​    单独更新该库 即可实现tensorflow lite之中的算子  包括：增加custom的op 以及 覆盖builtin的op   无需整体编译tf whl

```shell
bazel build --config opt //tensorflow/lite/python/interpreter_wrapper:tensorflow_wrap_interpreter_wrapper
```

构建结果为/bazel-bin/tensorflow/lite/python/interpreter_wrapper/_tensorflow_wrap_interpreter_wrapper.so

替换到~/anaconda3/envs/tf20/lit/python3.6/site-packages/tensorflow_core/lite/python/interpreter_wrapper/ 路径下

注意：对于tf 2.1 包路径是tensorflow_core 而不是之前的tensorflow

#### 增添custom op

tf 1.14  2.1 验证     

##### a. 编写算子源文件 

格式为：

```c++
// zero.cc
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {    // 需要放在custom命名空间
namespace zerof {
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {}                             
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {}
}  // namespace zerof
TfLiteRegistration* Register_ZEROF() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 zerof::Prepare, zerof::Eval};
  return &r;                     
}
}  // namespace custom
}  // namespace ops
}  // namespace tflite
```

如果加载模型报错：tensorflow_wrap_interpreter_wrapper.so: undefined symbol: _ZN6tflite3ops6custom30Register_EXTRACT_IMAGE_PATCHES

这是因为命名空间namespace custom 也许写成了builtin，混入了别的命名空间而无法被识别

在kernels路径下

##### b. BUILD文件添加规则

```python
cc_library(
    name = "builtin_op_kernels",
    srcs = [
        "activations.cc",
        "add.cc",
        "add_n.cc",
        "arg_min_max.cc",
....
         "zerof.cc",   ## 添加在这里 打包所有的算子
    ],
    hdrs = [
    ],
    copts = tflite_copts() + tf_opts_nortti_if_android() + EXTRA_EIGEN_COPTS,
```

##### c. 添加声明

在 (builtin_opkernels.h 后经验证不用)   custom_ops_register.h 添加

```c++
TfLiteRegistration* Register_ZEROS();
```

##### d. 注册算子

在 register.cc   register_ref.cc 添加

```c++
namespace custom {
TfLiteRegistration* Register_ZEROF();
}

BuiltinOpResolver::BuiltinOpResolver(){
AddCustom("Zerof", tflite::ops::custom::Register_ZEROF());  
    // Zerof 这里我们需严格使用大驼峰命名 例如: ExtractImagePatches
    // 在编译为解释器之后 算子名将被自动转化为下划线所谓snack格式 例如: extract_image_patches
    // 另注: Register_ZEROF这个名称没有上面两条的限制 随便定义 统一即可
}
```

##### e. polish编译验证流程

使用bazel build一次之后，tensorflow的源代码文件夹下已经包含了解释器的so文件

跳转到conda虚拟环境的tensorflow代码位置：~/anaconda3/envs/tf2.0/lib/python3.6/site-packages/tensorflow_core/lite/python/interpreter_wrapper

（注意：tf2.x以上在...site-packages/tensorflow_core/...    tf1.x则是在...site-packages/tensorflow/..）

为解释器so文件建立软连接

```shell
ln -s /home/gx/myproj/tensorflow/bazel-bin/tensorflow/lite/python/interpreter_wrapper/_tensorflow_wrap_interpreter_wrapper.so _tensorflow_wrap_interpreter_wrapper.so
```

每次更改算子kernel源代码 -> bazel build -> restart解释环境 例如jupyter notebook 即可使用更新的op算子

##### f. 算子增加属性

在tflite中custom op的属性和tf中custom op的属性不同，是因为保存模型的格式不同

tflite使用flexbuffer进行模型保存 是flatbuffer的一个精简子集

以extract_image_patches算子为例，tf中该算子的属性包括

```c++
 std::vector<int32> ksizes_;
 std::vector<int32> strides_;
 std::vector<int32> rates_;
 string padding_;
```

为增加四个属性，需要在custom op的kernel源代码中实现Init和Free方法

首先，在命名空间中定义一个结构体用来暂存各个属性

```c++
namespace extract_image_patches {
typedef struct {
  std::vector<int32> ksizes_;
  std::vector<int32> strides_;
  std::vector<int32> rates_;
  string padding_;
} TfLiteEIPOParams;
...
```

第二，在Init方法中从tflite的flexbuffer格式模型中读出属性值，存入上述结构体

```c++
// 加上相关include
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include <vector>
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/op_macros.h"


void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new TfLiteEIPOParams;//开辟一块空间存属性值
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);//属性存在模型的buffer里面
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();//参考flexbuffer官网文档 m可以理解为一个json
  
  //这里需要用AsTypedVector读出，如果用AsVector会失败，原因不明，可能flexbuffer做了奇怪的优化
  //注意：此处读出为vector其实不是cpp的std vector 而是flexbuffer的vector 
  //     所以还需通过AsInt32等方式获取为数值
  //     同样的AsString()之后需要用c_str()转化为cpp的格式
  data->rates_.push_back(m["rates"].AsTypedVector()[0].AsInt32());
  data->rates_.push_back(m["rates"].AsTypedVector()[1].AsInt32());
  data->rates_.push_back(m["rates"].AsTypedVector()[2].AsInt32());
  data->rates_.push_back(m["rates"].AsTypedVector()[3].AsInt32());

  data->ksizes_.push_back(m["ksizes"].AsTypedVector()[0].AsInt32());
  data->ksizes_.push_back(m["ksizes"].AsTypedVector()[1].AsInt32());
  data->ksizes_.push_back(m["ksizes"].AsTypedVector()[2].AsInt32());
  data->ksizes_.push_back(m["ksizes"].AsTypedVector()[3].AsInt32());

  data->strides_.push_back(m["strides"].AsTypedVector()[0].AsInt32());
  data->strides_.push_back(m["strides"].AsTypedVector()[1].AsInt32());
  data->strides_.push_back(m["strides"].AsTypedVector()[2].AsInt32());
  data->strides_.push_back(m["strides"].AsTypedVector()[3].AsInt32());
    
  data->padding_ = m["padding"].AsString().c_str();

  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<TfLiteEIPOParams*>(buffer);//最后释放空间
}
```

第三，在Prepare和Eval方法中，查寻该属性

```c++
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
...
auto* params = reinterpret_cast<TfLiteEIPOParams*>(node->user_data);
//其实属性被解释器读取后 放在node的user_data之中 读出并用TfLiteEIPOParams格式化
auto temp = params->ksizes_[0]; //使用属性值
...
}
```

##### g. TfLite  输出tensor维度修改

TfLite将模型的tensor统一用指针形式保存到context里面，对于一个op想输出指定维度的tensor，需要用到context类自带的ResizeTensor方法，参见源代码其他op的示例

需要特定的参数TfLiteIntArray* dims，指定resize目标tensor的维度

TfLiteIntArray结构包含int size和int data[] 依次指定即可，例如：

```c++
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
...
  TfLiteIntArray* ret = TfLiteIntArrayCreate(input->dims->size);
  ret->data[0] = 1;
  ret->data[1] = 2;
  ret->data[2] = 2;
  ret->data[3] = 1;

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(ret));
}
```



### 4.3.4 构建Android解释器aar包

jitbrain Android Studio(AS)（自带安卓skd ） +  ndk18b（验证可用 网上说必须15 垃圾）  +    bazel    tf2.1

配置tf Download a fresh release of clang : y   自配workspace 指定上面安好的 sdk ndk 路径

必要时修改配置文件.tf_configure.bazelrc： build --action_env ANDROID_NDK_API_LEVEL = 21 

```shell
bazel build --cxxopt='--std=c++11' -c opt        \
  --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a   \    # 两个arm基本包含大部分安卓手机
  //tensorflow/lite/java:tensorflow-lite
```

构建 .aar文件 结果保存在

bazel-genfiles/tensorflow/contrib/lite/java/tensorflow-lite.aar

在AS导入并在build gradle里写入依赖：implementation project(':tensorflow-lite')

#### 增添custom op

##### a. 编写算子源文件

直接将算子源文件 改扩展名为zerof.h  并放置在tensorflow/lite/java/src/main/native路径下

##### b. BUILD文件添加规则

```python
# *This includes all ops. If you want a smaller binary, you should copy and*
# *modify builtin_ops_jni.cc.  You should then link your binary against both*
# *":native_framework_only" and your own version of ":native_builtin_ops".*
# 如果要精简op集 进一步的减小tflite模型大小 参见下文c部分
cc_library(
    name = "native",
    srcs = [
        "builtin_ops_jni.cc",
    ],
    hdrs = [
        "zerof.h",    ## 头文件的build说明添加在这里 给builtin_op_jni.cc添加该算子
    ],
    copts = tflite_copts(),
    deps = [
        ":native_framework_only",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)
```

##### c. 注册算子

在builtin_op_jni.cc 文件内添加头文件声明

```c++
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/java/src/main/native/zerof.h"    #改成头文件 添加在这里
namespace tflite {
...
```

##### d. 精简op集

register.h 的 BuiltinOpResolver 类的构造方法  官方已经默认在register.cc中实现  

为了精简op集 可以自己overload该方法 

builtin_op_jni.cc 原始代码：

```c++
std::unique_ptr<OpResolver> CreateOpResolver() {  // NOLINT
  return std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver>(
      new tflite::ops::builtin::BuiltinOpResolver());
}
```

在原本的return里面 替换为：

```c++

std::unique_ptr<OpResolver> CreateOpResolver() {  // NOLINT
  return std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver>(
    new tflite::ops::builtin::BuiltinOpResolver::BuiltinOpResolver() {
      AddBuiltin(BuiltinOperator_ABS, Register_ABS());
      AddBuiltin(BuiltinOperator_MATRIX_SET_DIAG, Register_MATRIX_SET_DIAG());
    ... //删减一些算子
      AddCustom("Mfcc", tflite::ops::custom::Register_MFCC());
      AddCustom("Zerof", tflite::ops::custom::Register_ZEROF());
    } // end of BuiltinOpResolver(){}
  );  // end of return
} 
```

# 5. TensorFlow模型格式及转换

frozen固化模型是为了移动端部署实现的 但是lite也实现了该功能 

推荐方法简单处理后 将模型保存为 saved_model 并转化为lite模型

在tf2.x环境直接支持tf1.x模型运行：

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```



## 5.1 处理模型 保存为saved_model

```python
input_image_holder = tf.placeholder(tf.float32, [1,512,1360,3], name='input_image_holder') 
#设置输入占位符 类型 维度 node名称 
...
tf.saved_model.simple_save(sess,
	"./save_model_path",  #输出路径
	inputs={"input_image_holder": input_image_holder},  # node名 输入对象 
	outputs={"saturate_cast": output}) # 输出node名（根据实际情况） 输出对象
```

## 5.2 将saved_model 转化为lite模型

给lite添加custom op 分两种情况：

1. tensorflow含有该算子 而lite没有  

   ​		可以参照官网 使用tf算子库 后果是模型变得很大 官方还有白名单 有些op并不支持添加

   ​		可以自己在lite中实现该op  推荐

2. tensorflow不包含该算子 想在lite之中添加

   ​        需要先在官方tf库中 利用.so文件添加该算子  接着在lite的python解释器中添加  最后在移动端解释器添加

```python
tf.load_op_library('??/zerof.so')  ## 必须先导入custom op 否则无法识别 
converter = tf.lite.TFLiteConverter.from_saved_model('./save_model_path_zerof') # 导入
converter.allow_custom_ops = True  ## 事实上这里是：允许tf包含 但是lite不包含的op被打包进lite模型
# converter.post_training_quantize = True   ## 模型后处理量化开关
tflite_model = converter.convert()
open("./1.tflite", "wb").write(tflite_model)
```

## 5.3 pc机python环境导入并运行lite模型

```python
interpreter = tf.lite.Interpreter(model_path='/home/gx/myproj/generative_inpainting-master/1.tflite')  
# 导入模型创建解释器 如果custom op在kernels目录的某处没有注册 会提示不认识的符号
interpreter.allocate_tensors()  # 分配张量
input_details = interpreter.get_input_details()
print(str(input_details))
output_details = interpreter.get_output_details()
print(str(output_details))

input_shape = input_details[0]['shape'] 
input_data = np.array(np.random.random_sample(input_shape), dtype=np.int32) # 随机输入
print(input_data)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index']) # 取输出观察
print(output_data)
```

# 6. TfLite模型手动解析

## 6.1 flatbuffer和flexbuffer

### 6.1.1 简介及schema文件

flatbuffer是在高速场景下替代json的序列化开源库，因为是二进制格式，不需要像json建立诸多对象，存取速度很快

tflite模型存储格式为flatbuffer的精简版本flexbuffer  官方文档链接如下

https://google.github.io/flatbuffers/flatbuffers_guide_tutorial.html

flatbuffer需要由模板文件schema来定义，该文件详细定义了在二进制buffer中以怎样的步进读取数据，此处不赘述

tensorflowlite的模型存储模板schema文件在tensorflow源代码中可以找到：

<source_root>/tensorflow/lite/schema/schema.fbs

### 6.1.2 解析工具flatc

确保cmake已经正确安装 推荐官网二进制版本

克隆flatbuffer的源代码仓库git clone https://github.com/google/flatbuffers.git

```shell
cmake -G "Unix Makefiles" //生成MakeFile
make //生成flatc
make install //安装flatc
flat --version
```

解析tflite文件到JSON格式：注意文件名前--后的空格，注意strict json参数保证输出的json是带双引号的严格格式

```shell
./flatc -t schema.fbs -- my.tflite --strict-json
```

该模型my.tflite将被解析为my.json

```shell
./flatc -b schema.fbs my.json
```

该文件my.json将被反解析回my.lite模型

上述两个转化操作均会自动覆盖更新转化的目标文件

## 6.2 TfLite模型定义

TfLite模型的关键数据结构定义如下（部分省略）：

```json
Model
  operator_codes                     //列出用到的全部算子 对应的代码
    builtin_code 
  subgraphs                          //子图
    tensors                          //列出用到的所有张量（图的边）
      shape
      buffer                         //该张量参数存储使用到的buffer编号
      name                           //张量名 定义模型时用name命名 或者自动命名 netron等工具查看
      quantization              
    inputs                           //输入节点（node）编号
    outputs                          //输出节点编号
    operators                        //列出用到的所有节点
      opcode_index                   //该节点的算子代码序号 对应operator_codes 中序号
      inputs                         //该节点输入张量编号
      outputs                        //该节点输出张量编号
      ...                            //其他属性等
  buffers                            //参数仓库
  metadata                           // tf 2.x 添加这个描述 对应2.x的tflite解释器 不修改会出错
    name
    min_runtime_version
    buffer
```

例如需要手动修改移除某节点，流程如下：

a. 将tflite模型通过flatc转换成json格式

b. 在json文件中找到需要移除的节点（model subgraph operators xxx）把他的输入张量对接给他的下层节点，

c. 删除该operators防止悬空节点出现

d. 在operator_codes等处做相应修改

e. 使用flatc工具更新tflite模型

参见tflite_to_csv脚本 直接运行流程

## 6.3 TfLite解释器加载与解析

解释器加载tflite模型，放入内存只读区域，也有一些需要动态更新的buffer另开辟动态区域，解释器主要处理张量和节点两大内容

a. 对于tensor，首先解析为TfLiteTensor，并且将所有的TfLiteTensor整合为一张TfLiteContext表

​	在TfLiteContext中保存的是tensor的大小格式等信息和指向内存中加载模型真实张量的指针

b. 对于node，首先解析operators为TfLiteNode

​	并为之对应的匹配存在于TfLiteRegistration中的算子的kernel指针

至此完成了模型的加载

调试tflite的解释器，需要从subgraph中的PrepareOpsStartingAt等函数中使用

```c++
fprintf(stderr,"subgraphe allocatetensor start: %d of %d\n",execution_plan_index,execution_plan_.size());
```

等语句进行控制台调试

# 7 kernels修补

## 7.1 ResizeNearestNeighbor 增加中心对齐（align corner）功能

实际实现在kernels/internal/reference/reference_op.h中 line4364

原本函数秀了神bug：  // Align corners = true is not supported.然后在check之后没有写align corner功能

补充之

```c++
template <typename T>
inline void ResizeNearestNeighbor(
    const tflite::ResizeNearestNeighborParams& op_params,
    const RuntimeShape& unextended_input_shape, const T* input_data,
    const RuntimeShape& output_size_shape, const int32* output_size_data,
    const RuntimeShape& unextended_output_shape, T* output_data) {
  // Align corners = true is not supported.
  TFLITE_DCHECK(!op_params.align_corners);
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);

  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32 input_height = input_shape.Dims(1);
  int32 input_width = input_shape.Dims(2);
  int32 depth = MatchingDim(input_shape, 3, output_shape, 3);

  // The Tensorflow version of this op allows resize on the width and height
  // axis only.
  TFLITE_DCHECK_EQ(output_size_shape.FlatSize(), 2);
  int32 output_height = output_size_data[0];
  int32 output_width = output_size_data[1];

  // We use float to ensure agreement with the Tensorflow implementation.
  const float height_scale = static_cast<float>(input_height-1) / (output_height-1);
  const float width_scale = static_cast<float>(input_width-1) / (output_width-1);//增加align corner 功能

  const int col_offset = input_shape.Dims(3);
  const int row_offset = input_shape.Dims(2) * col_offset;
  const int batch_offset = input_shape.Dims(1) * row_offset;

  const T* input_ptr = input_data;
  T* output_ptr = output_data;
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      int32 in_y = std::min(static_cast<int32>((int)(y * height_scale +0.5)), //增加align corner 功能
                            input_height - 1);
      const T* y_input_ptr = input_ptr + in_y * row_offset;
      for (int x = 0; x < output_width; ++x) {
        int32 in_x = std::min(static_cast<int32>((int)(x * width_scale +0.5)),
                              input_width - 1);
        const T* x_input_ptr = y_input_ptr + in_x * col_offset;
        memcpy(output_ptr, x_input_ptr, depth * sizeof(T));
        output_ptr += depth;
      }
    }
    input_ptr += batch_offset;
  }
}

```



## 7.2 extractimagepatch 自定义

```c++
#include <algorithm> 

int pad_r_anchor = -1;
int pad_c_anchor = -1; 
for(int i=0; i<padding_row_num; i++){
    for(int j=0; j<padding_col_num; j++){
        int pad_r = pad_r_anchor + i*strides;
        int pad_c = pad_c_anchor + j*strides;
        int index = 0;
        for(int m=0; m<sizes; m++){
            for(int n=0; n<sizes; n++){
                int pointer_r = pad_r + m;
                int pointer_c = pad_c + n;
                if(pointer_r >= h||pointer_c >= w||pointer_r<0||pointer_c<0){   
                    for(int ic=0; ic<channels; ic++){
                        std::fill(out,out+channels,0);  //填零
                    } 
                }else{
                    for(int ic=0; ic<channels; ic++){
                        memcpy(out, in+(pointer_r*w+pointer_c)*channels,  channels* sizeof(float)); // 内存copy 速度快一些 要用到#include <algorithm> 
                    }
                }
                out+=channels;
            }
        }
    }
}
```

## 7.3 l2_norm（修改求均值的维度）

实际实现在kernels/internal/optimized/optimized_op.h中 line1369

原本函数求均值实在dim3上求得均值  而本模型要求在dim0 1 2上求均值

```c++

inline void L2Normalization(const tflite::L2NormalizationParams& op_params,  //你喵的参数params传进来用了么？！
                            const RuntimeShape& input_shape,
                            const float* input_data,
                            const RuntimeShape& output_shape,
                            float* output_data) {
  gemmlowp::ScopedProfilingLabel label("L2Normalization_gx");
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =  //864 dim0*dim1*dim2
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =  //5440 dim3
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  // fprintf(stderr,"\ntrailing_dim: %d, outer_size: %d, depth: %d\n",trailing_dim,outer_size,depth);
  // for (int i = 0; i < outer_size; ++i) {   //for 864        //原有错误写法
  //   float squared_l2_norm = 0;
  //   for (int c = 0; c < depth; ++c) { // for5440
  //     const float val = input_data[c];
  //     squared_l2_norm += val * val;
  //   }
  //   const float l2_norm = std::sqrt(squared_l2_norm);
  //   for (int c = 0; c < depth; ++c) {
  //     *output_data = *input_data / l2_norm;
  //     ++output_data;
  //     ++input_data;
  //   }
  // }
  for(int ch=0; ch<depth; ch++){    // for 5440 each channels
    float squared_l2_norm = 0;
    for(int p=0; p<outer_size; p++){  // for 864
      const float val = input_data[p*depth+ch];
      squared_l2_norm += val * val;
    }
    const float l2_norm = std::sqrt(squared_l2_norm);
    for(int p=0; p<outer_size; p++){
      output_data[p*depth+ch] = input_data[p*depth+ch] / l2_norm;
    }
  }
}

```

## 7.4 true_div自动转换问题

在tf中使用

```python
yi = tf.nn.conv2d_transpose(yi, raw_w,[1,128,170,96], strides=[1,rate,rate,1]) / 4.
```

最后的除四在转换为tflite的过程中，将可能会自动转化除进下一级的conv2d的kernels之中，也就是最后tflite的结果并没有div这个op







# 8 量化压缩和推断总览

函数调用关系总览

```c++
subgraph.cc        ../lite/core     子图 负责处理主要推断逻辑
    先调用节点init和prepare函数准备张量
    for node 遍历节点推断
        conv op invoke节点调用eval函数
        
conv.cc            ../lite/kernels  算子库
	prepare
	eval
		EvalFloat
			optimized_ops::Conv
		EvalHybrid
			optimized_ops::HybridConv
			
optimized_ops.h    ../lite/kernels/internal/optimized  优化算子负责实现计算逻辑
	Conv
		if DilatedIm2col
		else if Im2Col
		else 
		GEMM calculate //非压缩推断计算终点
	HybridConv
		if DilatedIm2col
		else if Im2Col
		else 
		NEON calculate //混合压缩推断计算终点

im2col_utlis.h     ../lite/kernels/internal/optimized  优化算子负责实现计算逻辑
	DilatedIm2col
	Im2Col
```



## 8.1 conv op

tflite的量化压缩原理参见英伟达的经典ppt，和ncnn类似，主要的压缩模式被称为混合压缩（全8bit压缩等不推荐）

也即是主要压缩卷积filter的参数

主数据是float32模式，输入conv算子

接着prepare中

首先判断是否需要hybrid混合压缩


```c++
 const bool is_hybrid =
      (input->type == kTfLiteFloat32 &&
       (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8));
```

第二使用AllocateTemporaryTensorsIfRequired 计算出data->need_im2col 即是否需要运行im2col优化

在eval中判断卷积类型

```c++
  switch (input->type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      if (filter->type == kTfLiteUInt8 || filter->type == kTfLiteInt8) {
        
        EvalHybrid<kernel_type>(context, node, params, data, input, filter,        // 混合卷积 输入float32 参数int8
                                bias, im2col, hwcn_weights, output);
      } else {
        EvalFloat<kernel_type>(context, node, params, data, input, filter, bias,   // 最正常的浮点卷积
                               im2col, hwcn_weights, output);
      }
      break;
    case kTfLiteUInt8:
      EvalQuantized<kernel_type>(context, node, params, data, input, filter,       // 输入也是uint8的全整数卷积
                                 bias, im2col, hwcn_weights, output);
      break;
    case kTfLiteInt8:
      EvalQuantizedPerChannel<kernel_type>(context, node, params, data, input,     // 输入是int8的全整数卷积
                                           filter, bias, output, im2col);
      break;
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           input->type);
      return kTfLiteError;
  }
```



最正常的浮点卷积：

调用了optimized_ops脚本中的conv，tflite把算子依据不同的使用场景分别写在不同的脚本之中

```c++
template <KernelType kernel_type>
void EvalFloat(TfLiteContext* context, TfLiteNode* node,
               TfLiteConvParams* params, OpData* data, TfLiteTensor* input,
               TfLiteTensor* filter, TfLiteTensor* bias, TfLiteTensor* im2col,
               TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
...
  switch (effective_kernel_type) {
    case kReference: {
      reference_ops::Conv(op_params, GetTensorShape(input),
                          GetTensorData<float>(input), GetTensorShape(filter),
                          GetTensorData<float>(filter), GetTensorShape(bias),
                          GetTensorData<float>(bias), GetTensorShape(output),
                          GetTensorData<float>(output), GetTensorShape(im2col),
                          GetTensorData<float>(im2col));
      break;
    }
    case kCblasOptimized:
    case kGenericOptimized: {  // 一般在这里 使用了通用优化的conv算子
      optimized_ops::Conv(op_params, GetTensorShape(input),
                          GetTensorData<float>(input), GetTensorShape(filter),
                          GetTensorData<float>(filter), GetTensorShape(bias),
                          GetTensorData<float>(bias), GetTensorShape(output),
                          GetTensorData<float>(output), GetTensorShape(im2col),
                          GetTensorData<float>(im2col),
                          cpu_backend_support::GetFromContext(context));
      break;
    }
    case kMultithreadOptimized: {
#ifdef TFLITE_WITH_RUY
      // See Register_CONV_2D: we should never be here when tflite_with_ruy
      // was enabled. We #if out this code in order to get the corresponding
      // binary size benefits.
      TFLITE_DCHECK(false);
#else
      const float* filter_data;
      if (data->need_hwcn_weights) {
        filter_data = GetTensorData<float>(hwcn_weights);
      } else {
        filter_data = GetTensorData<float>(filter);
      }
      multithreaded_ops::Conv(  // 如果支持多线程 会调用multithreaded_ops中的卷积
          *eigen_support::GetThreadPoolDevice(context), op_params,
          GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(filter), filter_data, GetTensorShape(bias),
          GetTensorData<float>(bias), GetTensorShape(output),
          GetTensorData<float>(output), GetTensorShape(im2col),
          GetTensorData<float>(im2col));
      break;
#endif
    }
  }
}
```

压缩使用的混合卷积：

```c++
template <KernelType kernel_type>
void EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                TfLiteConvParams* params, OpData* data, TfLiteTensor* input,
                TfLiteTensor* filter, TfLiteTensor* bias, TfLiteTensor* im2col,
                TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
                  
...
      optimized_ops::HybridConv(  // 调用了optimized_ops中的混合卷积函数
          op_params, scaling_factors_ptr, GetTensorShape(input),
          quantized_input_ptr_batch, GetTensorShape(filter), filter_ptr,
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output),
          GetTensorShape(im2col), im2col_ptr);
      break;
    }
  }
}
```

## 8.2 optimized_ops

在optimized_ops脚本中包含三个卷积conv

```c++
inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const uint8* input_data, const RuntimeShape& filter_shape,
                 const uint8* filter_data, const RuntimeShape& bias_shape,
                 const int32* bias_data, const RuntimeShape& output_shape,
                 uint8* output_data, const RuntimeShape& im2col_shape,
                 uint8* im2col_data, CpuBackendContext* cpu_backend_context)    // 全 int8卷积 input filter bias


inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data, const RuntimeShape& im2col_shape,
                 float* im2col_data, CpuBackendContext* cpu_backend_context)     // 全 float卷积 input filter bias
    
inline void HybridConv(const ConvParams& params, float* scaling_factors_ptr,
                       const RuntimeShape& input_shape,
                       const int8_t* input_data,
                       const RuntimeShape& filter_shape,
                       const int8_t* filter_data,
                       const RuntimeShape& bias_shape, const float* bias_data,
                       const RuntimeShape& output_shape, float* output_data,
                       const RuntimeShape& im2col_shape, int8_t* im2col_data)    // 混合卷积 int8-input int8-filter float-bias
```

其中im2col_data 是从conv op的eval函数就传进来的一个张量指针，一路传递到上述三个实际处理卷积逻辑的函数中，最终又传入im2col函数直接在内存上处理，并没有返回值

```c++
TfLiteTensor* im2col =
      data->need_im2col
          ? &context->tensors[node->temporaries->data[data->im2col_index]]
          : nullptr;
```

对于全float卷积：

```c++
inline void Conv(...) {
...
  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;                     // 判断是否需要处理空洞卷积
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;                  // 判断是否需要处理im2col优化
  if (need_dilated_im2col) {
    DilatedIm2col(params, float_zero_byte, input_shape, input_data,
                  filter_shape, output_shape, im2col_data);                          // 使用空洞卷积特别的im2col
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    Im2col(params, filter_height, filter_width, float_zero_byte, input_shape,
           input_data, im2col_shape, im2col_data);                                   // 使用无空洞卷积（dilatedrate=1）的im2col
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    // TODO(aselle): We need to make sure to not send im2col if it is not
    // needed.
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;                                                    // 不需要im2col优化 直接给gemm_input_data
    gemm_input_shape = &input_shape;
  }
...
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, gemm_input_data,       // 使用第三方gemm库加速矩阵乘法 
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
#endif  //  defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)
}
```



注意！！！对于混合卷积 tflite没有实现空洞卷积的判断，如果直接使用源代码会直接调用im2col引发崩溃

添加空洞卷积逻辑之后的修正代码，并传入正确的int8零值

```c++
inline void HybridConv(const ConvParams& params, float* scaling_factors_ptr,
                       const RuntimeShape& input_shape,
                       const int8_t* input_data,
                       const RuntimeShape& filter_shape,
                       const int8_t* filter_data,
                       const RuntimeShape& bias_shape, const float* bias_data,
                       const RuntimeShape& output_shape, float* output_data,
                       const RuntimeShape& im2col_shape, int8_t* im2col_data) {     
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  /////////////////////////////////  取出空洞卷积参数 并判断是否需要空洞卷积
  const int32 input_offset = params.input_offset;//
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  ////////////////////////////////////
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batch_size = input_shape.Dims(0);
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);

  const int8_t* gemm_input_data = nullptr;
  int num_input;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  /////////////////////////////////////
  if (need_dilated_im2col) {
    TFLITE_DCHECK(im2col_data);
    const int input_zero_point = 0x00; // 注意这里必须是0x00 也即是int8的零值 用于DilatedIm2col函数填充空位 否则会有意外错误
    TFLITE_DCHECK_GE(input_zero_point, 0);
    TFLITE_DCHECK_LE(input_zero_point, 255);
    DilatedIm2col(params, input_zero_point, input_shape, input_data,
                  filter_shape, output_shape, im2col_data);
    gemm_input_data = im2col_data;
    num_input = im2col_shape.FlatSize();
  } //  添加空洞卷积逻辑 将DilatedIm2col处理结果传给gemm_input_data
  else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    // symmetric quantization assumes zero point of 0.
    const int input_zero_point = 0;

    Im2col(params, filter_height, filter_width, input_zero_point, input_shape,
           input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    num_input = im2col_shape.FlatSize();
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    num_input = input_shape.FlatSize();
  }

  // Flatten 4D matrices into 2D matrices for matrix multiplication.
  // Flatten so that each filter has its own row.
  const int filter_rows = filter_shape.Dims(0);
  const int filter_cols = FlatSizeSkipDim(filter_shape, 0);

  // In MatrixBatchVectorMultiplyAccumulate, each output value is the
  // dot product of one row of the first matrix with one row of the second
  // matrix. Therefore, the number of cols in each matrix are equivalent.
  //
  // After Im2Col, each input patch becomes a row.
  const int gemm_input_cols = filter_cols;
  const int gemm_input_rows = num_input / gemm_input_cols;
  const int output_cols = output_shape.Dims(3);
  const int output_rows = FlatSizeSkipDim(output_shape, 3);
  TFLITE_DCHECK_EQ(output_cols, filter_rows);
  TFLITE_DCHECK_EQ(output_rows, gemm_input_rows);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_cols);

  // MatrixBatchVectorMultiplyAccumulate assumes that each row of the second
  // input matrix has its own scale factor. This code duplicates the scale
  // factors for each row in the same batch.
  const int rows_per_batch = gemm_input_rows / batch_size;
  for (int i = gemm_input_rows - 1; i >= 0; --i) {
    scaling_factors_ptr[i] = scaling_factors_ptr[i / rows_per_batch];
  }
  tensor_utils::ZeroVector(output_data, output_rows * output_cols);
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(    // 注意这里不是使用gemm来实现矩阵运算 而是使用neon逻辑 在tensor_utils脚本中
      filter_data, filter_rows, filter_cols, gemm_input_data,
      scaling_factors_ptr, /*n_batch=*/gemm_input_rows, output_data,
      /*result_stride=*/1);
  AddBiasAndEvalActivationFunction(output_activation_min, output_activation_max,
                                   bias_shape, bias_data, output_shape,
                                   output_data);        // 额外处理bias和激活函数
}
```



## 8.3 im2col_utlis.h

这个脚本处理im2col优化的实际逻辑 在optimized_ops命名空间下

主要包含三个函数：

```c++
inline void ExtractPatchIntoBufferColumn(const RuntimeShape& input_shape, int w,
                                         int h, int b, int kheight, int kwidth,
                                         int stride_width, int stride_height,
                                         int pad_width, int pad_height,
                                         int in_width, int in_height,
                                         int in_depth, int single_buffer_length,
                                         int buffer_id, const T* in_data,
                                         T* conv_buffer_data, uint8 zero_byte)
void DilatedIm2col(const ConvParams& params, uint8 zero_byte,
                   const RuntimeShape& input_shape, const T* input_data,
                   const RuntimeShape& filter_shape,
                   const RuntimeShape& output_shape, T* im2col_data)  // 自己实现全部逻辑 不使用ExtractPatchIntoBufferColumn
void Im2col(const ConvParams& params, int kheight, int kwidth, uint8 zero_byte,
            const RuntimeShape& input_shape, const T* input_data,
            const RuntimeShape& output_shape, T* output_data)         // 调用ExtractPatchIntoBufferColumn 处理每个col
```





# 9 tflite的c++构建和嵌入式部署

## 9.1 构建路径总览

原tensorflow的bazel构建文件里面提供了几处关于tflite解释器的构建target：

共享库 libtensorflowlite.so 主要由 framework 和 builtin_ops 两个target组成

```shell
A. lite根目录下的c++共享库
bazel build //tensorflow/lite:libtensorflowlite.so
            依赖：//tensorflow/lite:framework               <--
                 //tensorflow/lite/kernels:builtin_ops     <--
                 
B. lite示例代码中直接将源代码和共享库所需的framework以及builtin_ops一起构建为可执行文件
bazel build //tensorflow/lite/examples/minimal:minimal
            源文件：minimal.cc
            依赖：//tensorflow/lite:framework               <--
                 //tensorflow/lite/kernels:builtin_ops     <--
                 
C. python解释器封装
bazel build //tensorflow/lite/python/interpreter_wrapper:tensorflow_wrap_interpreter_wrapper
            依赖：//tensorflow/lite/python/interpreter_wrapper:interpreter_wrapper_lib
                  依赖：//tensorflow/lite:framework             <--
                       //tensorflow/lite/kernels:builtin_ops   <--
                       ...
                ...
 
 D. 安卓aar打包
 bazel build //tensorflow/lite/java:tensorflow-lite
            依赖：//tensorflow/lite/java:tensorflowlite
                  依赖：//tensorflow/lite/java:tensorflowlite_native
                       源文件：//tensorflow/lite/java:libtensorflowlite_jni.so
                             依赖：//tensorflow/lite/delegates/nnapi/java/src/main/native
                                  //tensorflow/lite/java/src/main/native
                                       :native
                                            源文件：builtin_ops_jni.cc
                                            依赖：//tensorflow/lite/kernels:builtin_ops   <--
                                                 :native_framework_only
                                                      源文件：...
                                                      依赖：//tensorflow/lite:framework   <--
```



## 9.2 cmake构建共享库和业务逻辑代码

为了给嵌入式平台使用tflite解释器和模型，以安卓为例，可以使用两类方法：

A. 直接将解释器封装为aar包，参见4.3.4内容，由安卓studio导入并使用（官方发布的标准tflite可以自动下载，然而自己添加op修改过的tflite未尝试成功）

B. 将tflite解释器封装为so文件，并在pc平台由传统cmake构建为二进制文件（目的是验证业务逻辑代码和模型执行有效性）。注意这里也可以采用bazel来进行构建打包验证，考虑到构建方式通用性，还是推荐选用cmake的方式来构建

### 9.2.1 cmake工程文件结构

```
├── build                    // im1_14_myatt.tflite 等模型文件和输入输出图片放在这里
├── include
|     ├──flatbuffers         // flatbuffers的include文件
|     └──tensorflow          // lite的include文件
├── lib                      // libtrensorflowlite.so 等共享库
├── src                      // 业务逻辑代码
└── CmakeLists.txt
```

flatbuffers可以参见6.1 下载git 仓库并取出 include文件夹

lite的include文件可以通过以下shell命令取出并解压到上述文件夹

```shell
cd tensorflow/tensorflow
find ./lite -name "*.h" | tar -cf headers.tar -T -
```

libtrensorflowlite.so通过 9.1的A路径构建：

```shell
bazel build //tensorflow/lite:libtensorflowlite.so --fat_apk_cpu=x86_64,arm64-v8a,armeabi-v7a --cxxopt="-std=c++11"
```

cmake文件为：

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_tflite)
set(CMAKE_CXX_STANDARD 14)
# 添加opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# 添加头文件 lite和flatbuffers相关
set(INC_DIR ./include)
include_directories(${INC_DIR})
# 添加共享库 tflite解释器
set(LINK_DIR ./lib)
link_directories(${LINK_DIR})
link_libraries(tensorflowlite )

add_executable(my_tflite ./src/minimal.cc)
target_link_libraries(my_tflite tensorflowlite  ${OpenCV_LIBS})
```

示例业务逻辑代码为：

```c++
#include <cstdio>
#include <ctime>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <opencv2/opencv.hpp>

using namespace tflite;
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
// std::string model_file = "./im1_14_myatt.tflite";
std::string model_file = "./im2_myatt.tflite";
std::string image_file = "./input.png";

cv::Mat image = cv::imread(image_file.c_str());
image.convertTo(image, CV_32FC3); // cpp read to float32 and 3 channels

std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;

tflite::InterpreterBuilder(*model, resolver)(&interpreter);

TfLiteTensor* input_tensor     = nullptr;
TfLiteTensor* output_image     = nullptr;

interpreter->AllocateTensors();

input_tensor = interpreter->tensor(interpreter->inputs()[0]);
interpreter->SetNumThreads(1);

// save opencv mat data into tflitetensor data.f
float* dst = input_tensor->data.f;
const int row_elems = image.cols * image.channels();
for (int row = 0; row < image.rows; row++) {
    const float* row_ptr = image.ptr<float>(row);  // use float, not uchar
    for (int i = 0; i < row_elems; i++) {
        dst[i] = row_ptr[i];
    }
    dst += row_elems;
}
// run inference
clock_t start = clock();
interpreter->Invoke();
clock_t end = clock();
cout << " invoke time:" << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;

// get output 
output_image = interpreter->tensor(interpreter->outputs()[0]);
// output_image = input_tensor;
cv::Mat OUT(output_image->dims->data[1],output_image->dims->data[2],CV_32FC3,output_image->data.f);
// swap channel012 to 210(this model need)
for (int i = 0; i < OUT.rows; ++i) {
    for (int j = 0; j < OUT.cols; ++j) {
        Vec3f& pix = OUT.at<Vec3f>(i, j);
        std::swap(pix[0], pix[2]);
    }
}
cv::imwrite("./output_iamge.png", OUT);
return 0;
}
```

构建和运行：

```shell
cd ./build
cmake ..
make
./my_tflite
```





### 9.2.2 bazel处理

如果非要使用bazel，则构建方式和可能遇到的问题如下：

构建时为了验证业务逻辑需要对opencv进行导入，方法如下：

目录结构

```
├── WORKSPACE
├── opencv.BUILD
├── main.cc
└── BUILD
```

WORKSPACE

```python
workspace(name = "bazel_test")
new_local_repository(
    name = "opencv",
    path = "/home/xxx/xxx/opencv/install",
    build_file = "opencv.BUILD",
)
```

opencv.BUILD

```python
cc_library(
    name = "opencv",
    srcs = glob(["lib/*.so*"]),
    hdrs = glob(["include/**/*.hpp"]),
    includes = ["include"],
    visibility = ["//visibility:public"], 
    linkstatic = 1,
)
```

BUILD

```python
cc_binary(
    name = "bazel_test",
    srcs = ["main.cc"],
    deps = [
        "@opencv//:opencv",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:builtin_ops",  
        # 注意：这里没有显式地写明依赖flatbuffers 是因为builtin_ops -> builtin_op_kernels -> @flatbuffers之中已经引用了 
    ],
)
```

 main.cc

```c++
#include <opencv2/opencv.hpp>
int main(int argc, char *argv[]) {
    cv::Mat img = cv::imread("/home/alan/1.jpg");
    std::cout << "Resolution: " << img.rows << " x " << img.cols << std::endl;
    return 0;
}
```

bazel最终构建方式为：bazel build //xxx/xxx:bazel_test



















































