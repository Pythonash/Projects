---
layout: post
section-type: post
title: Object detection (물체 탐지) You Only Look Once (YOLO, 욜로) 리뷰 및 코드 구현
category: paper
tags: [ 'deep learning','machine learning','data science','paper' ]
---

안녕하세요, Pythonash 입니다.

요즘 제가 느끼는 건데, 본업(연구)보다 이런 딥러닝 관련한 공부가 너무 재미있어서... 큰일이에요 ㅋㅋ.

예전에 연구 프로젝트 계획서 제출할 때, 딥러닝을 이용한 물체 탐지에 대해 더욱 자세하게 알아보다가 YOLO를 알게 되었는데요.

그때만 해도 YOLO의 버전이 3까지 나왔는데, 지금은 또 모르겠네요. 앞으로도 YOLO의 자매즈들을 한번 포스팅 해보겠습니다.

그래서 오늘 할 것은 바로, YOLO 첫 번째 버젼인 **You only look once (YOLO)**의 간단한 논문 리뷰와 코드 구현해보기 입니다.

처음에는 다들 욜로, 욜로 하길래 뭐지?? 싶었는데 You only live once가 아니라 look이라는 군요 ㅎ...

아무튼 서론이 길었는데 얼른 시작하겠습니다.

---

참고로 저는 코드 구현을 tensorflow와 keras를 이용해 구현하고 있습니다.

보통 연구목적으로 pytorch가 많이 사용되어온건 알지만, 그래도 이 글을 보시는 유저분들은 대부분 tensorflow와 keras가 아마 더 익숙하시지 않을까 생각합니다.

물론, 저도 마찬가지기 때문에 ㅎ...

진짜 **시작**하겠습니다.

# 목차

<a id="toc"></a>
- [1. YOLO의 구조](#1)
    - [1.1 YOLO 원리](#1.1)
    - [1.2 YOLO Design](#1.2)
    - [1.3 YOLO 구조 정리](#1.3)
- [2. Code 구현](#2)
    - [2.1 데이터셋](#2.1) 
    - [2.2 모델구현](#2.2)
    - [2.3 훈련함수 작성](#2.3)
    - [2.4 GAN 훈련](#2.4)
- [3. Review ](#3)
- [4. References](#4)

<a id="1"></a>
# YOLO의 구조

기존의 물체 탐지는 분류(classification)를 한다는 접근으로 이루어졌었는데 이 논문에서는 **바운딩 박스와 class 확률에 대한 회귀(regression) 문제로 접근**합니다.

단일의 네트워크로 이루어진 구조로 바운딩 박스를 예측하고 클래스 확률을 추정할 수 있으며, 탐지 성능에 대한 end-to-end 최적화가 가능하다고 합니다.

더불어, 이 단일화된 구조가 빠른 것은 물론이고 실시간으로 초당 45 프레임들의 이미지를 처리할 수 있다고 합니다.

더 작은 버젼인 Fast YOLO의 경우에는 초당 155 프레임들을 다루고, 다른 실시간 탐지기들보다 두배나 되는 mAP를 달성한다고 합니다.

기존 모델 중에서 **Deformable parts models (DPM)**는 전체 이미지에 고르게 분류기들을 위치시키고 작동시키는 **sliding window approach**를 사용합니다.

이보다 더 최근 모델인 **R-CNN**은 **region proposal method**를 사용하는데, 이는 **1) 바운딩 박스를 처음에 생성**하고 그때 생성된 바운딩 박스에 분류기를 돌립니다.

이후에, **2) 바운딩 박스를 정제**하고, **중복 탐지 제거하는 등의 추가 전처리**를 하게 됩니다.

그런데 이러한 **복잡한 설계**는 각 모듈(모델을 구성하는 요소 등)을 **하나씩 분리해서 훈련**시켜야 하기 때문에 **느리고 최적화 하기도 어렵다**고 합니다.

그래서 본문에서는 기존의 모델들 처럼 '분류'가 아닌 **'회귀'로 접근**하겠다는 건데요, 이미지에 있는 **픽셀로부터 바운딩 박스의 좌표와 클래스를 예측**하겠다는 겁니다.

그러면 이 시스템을 이용해서, 우리는 이미지를 봤을 때 **물체가 어디 있는지**, 그게 **무엇인지 한번만 보면 된다**는 컨셉(즉, 말 그대로 **You only look once, YOLO**)이 되겠습니다.

이런 단일화된 구조는 기존의 물체 탐지 방법보다 **몇 가지 장점**을 지니고 있습니다.

### **1. YOLO is extremely fast.**

먼저, 물체 탐지의 틀을 회귀 문제로 바꿨기 때문에 **복잡한 구조가 필요하지 않다**는 것 입니다.

심지어 배치로 학습시키지 않아도, 초당 45 프레임, fast YOLO에서는 초당 150 프레임(Titan X GPU를 사용헀을 때인데, 이게 출고가가 찾아보니까 거의 $ 1,000..)의 성능을 냅니다.

이 의미는 비디오에도 **실시간으로 잘 작동**할 수 있다는 것입니다.

그리고 다른 실시간 시스템들보다 성능도 좋습니다.

이것과 관련해서 논문 저자들이 소개해 놓은 **[홈페이지] (http://pjreddie.com/yolo/)**를 들어가시면 실시간 비디오로 객체 탐지가 어떻게 이루어지고 있는지 보실 수 있습니다.

### **2. YOLO reasons globally about the image when making predictions.**

다른 물체 탐지 모델(DPM, R-CNN 같은)과는 다르게 **YOLO는 전체 이미지를 보고 훈련**합니다.

그래서 잠재적으로는 **클래스 뿐만아니라 외형과 같은 문맥상의 정보도 캐치**할 수 있는 겁니다.

### **3. YOLO learns generalizable representations of objects.**

YOLO는 물체의 **일반화된 표현을 학습**할 수 있어서 **다른 주제나 예상치 못한 물체(input)에도 잘 작동**할 수 있습니다.

하지만, **단점도 존재**하는데 **작은 물체와 같은 것들은 정확하게 추정하는데 어려움**을 겪고 있습니다.

그래서 **빠른 탐지가 가능한 기능 vs 작은 물체의 정확한 추정 간의 트레이드 오프**를 이후의 실험에서 연구해보겠다고 합니다.

참고로 학습 코드와 테스팅 코드는 모두 **오픈 소스**여서 사전 학습된 모델도 다운로드 가능하다고 합니다.

---
<a id="1.1"></a>
## YOLO 원리

YOLO는 물체 탐지의 분리된 구성 요소를 하나의 단일 모델로 통합했고, 한 이미지에서 모든 바운딩 박스와 클래스들을 동시에 예측합니다.

이것이 바로 위에서 언급한 "YOLO의 globalzation" 즉, 이미지와 물체에 대해 전체적으로 잘 설명할 수 있는 이유 입니다.

탐지를 하는 원리는 다음과 같습니다.

### **1. 이미지를 S x S 격자로 나눈다.**

만약 물체의 중심부분이 어떤 격자 셀의 중심에 있으면 그 격자 셀은 해당 물체를 탐지하는 중추가 됩니다(논문에서는 be responsible for detecting that object).

### **2. 각 격자셀은 바운딩 박스들과 신뢰점수(confidence score)를 예측한다.**

여기서 신뢰점수는 **1) 물체를 포함하는 박스를 얼마나 신뢰할 수 있는지**, 그리고 모델이 생각했을 때 **2) 그 박스가 얼마나 정확하게 예측**을 하는지를 반영하게 됩니다.

그리고 이것을 **Pr(object) x IOU**로써 정의하고 있습니다. 

> 만약 셀 안에 물체가 없다면 신뢰점수는 0이 될 것입니다.

한편으로는 이 신뢰점수를 **Intersection Over Union (IOU)와 같게 하는 것을 목표**로 하고 있습니다(즉, 물체를 정확하게 탐지하고자 함).

### **3. 각각의 바운딩 박스는 **x, y, w, h 그리고 신뢰도로 구성**되어 있다.**

x, y는 격자 셀 테두리에 **상대적인 박스의 중심을 나타내는 좌표**입니다.

넓이와 높이는 전체 이미지에 **상대적인 값으로 예측**됩니다.

마지막으로 **신뢰도는 IOU**를 나타냅니다.

### **4. 각 격자셀은 또한, class에 대한 조건부 확률을 추정한다.**

![image](https://user-images.githubusercontent.com/91790368/153373379-7bcc8222-5659-473d-b62e-4a0d09c75d59.png)

여기서 **Pr(Class(i)|Object)는 물체가 주어졌을때 그것이 각 클래스에 속할 확률을 나타내는 조건부 확률**인데, 본 논문에서는 **격자 셀 당 하나의 클래스 확률 집합을 추정**합니다.

다시 말해서, 하나의 격자 셀은 해당 물체가 어떠 어떠한 클래스들에 속할지에 대한 **하나의 확률 집합을 계산**한다는 것 입니다.

예를 들면, 어떤 **격자 셀 안에 이미지 하나가 잡혔는데 이게 강아지에 속할 확률, 고양이에 속할 확률, ..., 사람에 속할 확률 각각 하나씩 확률을 추산**한다는 것 입니다.

아무튼 이 조건부 확률을 이용해서 다음과 같은 식을 만들 수 있습니다.

![image](https://user-images.githubusercontent.com/91790368/153373245-cdb5bfa7-1831-4621-bce4-5a9aacf0e1b3.png)

이 식을 통해 각각의 **바운딩 박스에서 특정 클래스에 대한 확률 정보**를 얻을 수 있습니다.

그리고 이 점수들은 **박스 안에 나타나는 클래스의 확률과 얼마나 그 박스가 물체에 잘 맞는지를 반영**합니다.

---

<a id="1.2"></a>
## YOLO Design


일단 **YOLO는 합성곱 네트워크로 구성**되어 있고, 이를 **PASCAL VOC 탐지 데이터셋에서 평가**를 했다고 합니다.

그리고 **GoogLeNet으로부터 영감을 받아 YOLO를 구성**했는데, 사전훈련 작업으로 ImageNet 데이터셋으로 사전 훈련 시키기도 하고 등등...많은 작업을 했더군요.

이 모델의 구조는 논문에 **Figure 3: The Architecture**에 보면 보실 수 있습니다.

아무튼 이런 저런 사전 작업을 거친 후에는 입력 이미지의 사이즈를 **(224 x 224) 에서 (448 x 448)로 늘려줬는데** 그 이유는 물체 탐지에 있어서 **fine-grained한 시각 정보를 넣기 위함**이라고 합니다.

그런데 여기에서, **fine-grained는 잘 걸러진, 이미지로 치면 보다 범주화가 잘된 그룹들로 묶인 이미지? 같은 느낌**인데 논문에 쓰인 느낌으로는 그냥 **탐지 작업에 더 많은 정보를 주기 위함**으로 이해했습니다.

Anyway, 이 밖에도 **바운딩 박스의 넓이와 높이**를 전체 이미지의 넓이와 높이로 각각 나눠서 **정규화(0과 1사이)하고**, Leaky Rectified linear activation (**Leaky ReLU activation**)을 사용한다고 합니다.

특이한것은 최적화 하기 쉽다는 이유로 **최적화할 에러가 오차제곱합(sum-squared error)**인데, 이는 평균적인 **정밀도를 최대화 한다는 목적에 부합하는 것은 아닙니다.**

오히려 분류 오차와 같이 **국지적인 오차에만 가중치를 둬서 이상적이지 않습니다.**

또한, 전체 이미지에 많은 격자셀들은 **어떠한 물체도 포함하지 않는데** 이는 물체를 포함하고 있는 **셀의 가중치를 억눌러 신뢰점수를 0으로 만들게끔 합니다.**

결과적으로는 이러한 것들이 **모델을 불안정하게 만들고 학습을 발산 시켜버리게 만드는 원인**이 됩니다.

그래서 이러한 것들을 방지하기 위해, 오히려 **바운딩 박스의 좌표 예측을 담당하는 손실(loss)을 증가**시켜버리고 **물체를 포함하고 있지 않은 박스의 신뢰 점수에 대한 손실을 낮춰**버립니다.

한마디로, 규제와 같이 **학습을 조금 어렵게 만들어서 오히려 모델을 더 정확하게 훈련**시키는 식으로 생각하면 될 것 같습니다.

종합하면 다음과 같은 손실 함수를 최적화하는데, 본문에서는 **multi-part loss function**으로 정의하고 있습니다.

(사진 올리기)


여기서 주목하실 것은 격자셀 안에 **물체가 존재할 때에만 분류 오차에 규제를 가한다**는 것과

**가장 높은 IOU를 갖는 bounding box를 "responsible"한 예측기로 간주**하는데, 이것이 있을때에만 바운딩 박스 좌표에러에 패널티를 가합니다.

이후의 내용에는 64개의 배치, 0.9 momentum과 0.0005 decay 등 학습 파라미터에 대한 설명이 나오고, 학습률 스케쥴링을 에폭마다 어떻게 했는지, 또 과적합을 방지하기 위해 드롭아웃과 data augmentation을 진행 한 것 등등에 대한 설명이 나와 있습니다.



<a id="1.3"></a>
## YOLO 구조 정리

1. 물체탐지를 분류(classification)가 아닌 회귀(regression)로 접근한다.

2. 단일화된 네트워크로 구성되어 있고, 이미지에 대한 전체적인 정보를 학습한다.

3. 그렇기 때문에, 최적화가 간편하고 다른 물체들에 대한 적용 가능성도 높다.


---

<a id="2"></a>
## Code 구현

코드를 구현하고 결과를 살펴보겠습니다.

정확한 코드는 제 [깃허브](https://github.com/Pythonash/Projects/blob/Brain/GAN%EC%9D%84%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%EC%BB%AC%EB%9F%AC%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EA%B5%AC%ED%98%84%20.ipynb)를 클릭하시면 보다 상세하게 보실 수 있습니다.

<a id="2.1"></a>
## 데이터셋

데이터셋의 경우 케라스에서 제공하는 CIFAR100 데이터셋을 이용했습니다.

보다 선명한 사진으로 하고 싶었는데... 그래도 결국 본질이 중요한거니깐요... 일단 진행하겠습니다.


먼저 데이터셋을 불러오겠습니다. 그리고 나서 이 이미지가 어떻게 생겼는지 보겠습니다.
<pre>
<code>

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
plt.imshow(x_train[0])

</code>
</pre>

[실행결과]

![image](https://user-images.githubusercontent.com/91790368/152830417-0d6996f1-3c65-46a9-9a30-fa1a834dbf57.png)

아주 우직한 소 한마리가 있네요. 근데 사진이 흐려서 잘 보이진 않습니다 ㅋㅋ.. 소 맞겠죠?

그 다음에는 데이터셋들을 합쳐줄 겁니다. 원래 불러올 때에는 훈련용, 테스트용 데이터셋을 불러왔는데 이걸 하나로 합쳐서 6만개 데이터셋으로 만들어 줄려고 합니다.

<pre>
<code>

df = np.concatenate([x_train, x_test])
print(df.shape)

</code>
</pre>

[실행결과]

<pre>(60000, 32, 32, 3)
</pre>

데이터가 잘 합쳐진 것 같습니다. 각 숫자가 의미하는 것은 데이터 6만개, 32 x 32 크기의 사진, RGB 세 개의 채널을 가진 컬러이미지 라는 뜻 입니다.

이제 모델을 구축해보겠습니다.


<a id="2.2"></a>
## 모델구현

<pre>
<code>

size = 100

generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8 * 8 * 128, input_shape = [size]),
    tf.keras.layers.Reshape([8, 8, 128]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(64, kernel_size = 5, strides=2, padding='same', activation ='elu', kernel_initializer = 'he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(32, kernel_size = 5, strides=2, padding='same', activation = 'elu',kernel_initializer = 'he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2DTranspose(3, kernel_size = 5, strides=1, padding='same', activation = 'elu',kernel_initializer = 'he_normal')
])

discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, kernel_size = 5, strides =2, padding='same', activation = tf.keras.layers.LeakyReLU(0.3),
                          input_shape=[32,32,3], kernel_initializer = 'he_normal'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, kernel_size = 5, strides=2, padding='same',
                          activation = tf.keras.layers.LeakyReLU(0.3), kernel_initializer = 'he_normal'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

gan = tf.keras.models.Sequential([generator, discriminator])
gan.summary()

</code>
</pre>

일단 제일 위에부터 설명드리겠습니다.

**size**는 일종의 seed number와 같이 생각하시면 됩니다. 

> 생성자 모델의 입력과 대응되는 부분인데, 얼만큼의 사이즈를 넣어줄지 결정하는 겁니다.

> 하이퍼 파라미터처럼 여러 값을 시도해보실 수 있습니다.

**generator**는 생성자 모델을 구축한 겁니다.

> 중간 중간 BatchNormalization을 해줬고, 전치합성곱을 쌓아줬습니다. 

> 폭주를 막기위해서 드롭아웃층과 효율적인 학습을 위해 he_normal을 이용한 ReLU함수의 변종을 사용했습니다.

**discriminator**는 판별자 모델을 구축한 겁니다.

> 생성자모델은 출력층에서 (32,32,3)의 차원을 갖는 데이터를 뱉어낼 겁니다. 그러면 다시 (32,32,3)의 차원을 갖는 입력층을 판별자에 쌓아줍니다.

> 그리고 역시 BatchNormalization과 드롭아웃을 추가하고 마지막에 이 데이터를 일렬로 펼치는 Flatten과 마지막에 진짜인지 가짜인지를 가려낼 시그모이드 활성화 유닛을 쌓아줍니다.

**gan**은 생성자와 판별자 모델을 합쳐준 겁니다.

> 이제 생성자가 랜덤 벡터로부터 입력을 받아서 아무 이미지나 뱉어내면, 판별자가 그 이미지를 집어삼킨후 진짜인지 가짜인지를 판별할 겁니다.

> 그러면 그 판별자를 보고, 생성자는 더 잘 속이기 위해 점점 진짜와 같은 이미지를 만들어 낼 겁니다.

[실행결과]

<pre>Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
sequential_1 (Sequential)   (None, 32, 32, 3)         1086787   
_________________________________________________________________
sequential_2 (Sequential)   (None, 1)                 911617    
=================================================================
Total params: 1,998,404
Trainable params: 1,964,932
Non-trainable params: 33,472
_________________________________________________________________
</pre>

참고로 여기서 Non-trainable params라고해서 업데이트 되지 않는 파라미터들이 있는데 이는 BatchNormalization으로 인해 생기는 파라미터들 입니다.

그럼 이제 이미지 데이터를 Min-Max 스케일링을 해주고, 다시 그 값을 -1과 1사이의 값으로 조정해줄 겁니다.

왜냐하면, 원래 이 모델은 "심층 합성곱 GAN" 이라는 것을 골자로 파라미터를 제가 바꾼 것인데 기존 모델에서는 생성자 모델의 출력층에 하이퍼볼릭 탄젠트 함수(-1과 1사이의 값 출력)를 썼기 때문입니다.

근데 저는 elu로 해도 이게 잘 작동하는지 보기위해 생성자 출력층에 그냥 elu 활성화 함수를 사용했는데, 이걸 참고하시는 분들은 굳이 -1과 1사이로 값을 조정해주지 않고 정규화(Min-Max)만 해주셔도 됩니다.

<pre>
<code>

image = df/255
imgae = image * 2. -1.

</code>
</pre>


## 데이터셋 적재

이제 이 데이터셋들을 tensor로 바꾸고 배치로 꺼내서 학습시켜주기 위해 다음과 같이 작업합니다.

<pre>
<code>

batch_size = 30
dataset = tf.data.Dataset.from_tensor_slices(image)
dataset = dataset.shuffle(2022)
dataset = dataset.batch(batch_size, drop_remainder = True).prefetch(1)

</code>
</pre>

배치 사이즈는 30으로 일단 해줬습니다. 근데 다르게 하시더라도 뒤에 drop_remainder = True로 지정해주시면 괜찮습니다.

또, 여기서 prefetch(1)을 하게 되면 더욱더 빠르게 학습이 가능하다고 알고 있습니다.

학습할때 배치마다 하나씩 데이터를 준비시키는 원리라고 하더라구요.

이제 데이터셋은 준비가 됐고, 다시 모델 구축을 마무리 해보겠습니다.

## 판별자 프리징

GAN의 경우에는 기존 신경망 학습과는 다르게 두 단계에 걸쳐 진행이 됩니다.

먼저 생성자가 데이터를 생성하고 그걸 판별자가 학습합니다.

그 다음 gan을 통해 모델의 가중치를 생성자가 받아 업데이트를 하는 방식인데요, 말하자면 생성자와 판별자는 동시에 학습이 되면 안된다는 겁니다.

또한, 훈련 함수르 따로 작성해줄건데 그때 판별자의 가중치를 얼리고 녹이고를 반복할 것이라 먼저 얼려줍니다.

<pre>
<code>

discriminator.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')
discriminator.trainable = False

</code>
</pre>

## GAN 컴파일

이제 본격적으로 학습에 들어가기 전에 GAN을 컴파일 해줍니다. 그러면 모델의 구조가 이렇구나~ 하고 인식하게 될겁니다.


<a id="2.3"></a>
## 훈련함수 작성

앞서 말씀드렸듯이, GAN은 조금 특별한 방식으로 훈련이 됩니다.

따라서 기존의 fit()을 호출해서 학습시키지 않고, 따로 훈련함수를 만들어 주겠습니다.

<pre>
<code>

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
  generator, discriminator = gan.layers
  for epoch in range(n_epochs):
    print("Epoch {}/{}".format(epoch + 1, n_epochs))
    for X_batch in dataset:
      noise = tf.random.normal(shape = [batch_size, codings_size])
      generated_images = generator(noise)
      generated_images = tf.cast(generated_images, tf.float64)
      X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
      y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
      discriminator.trainable = True
      discriminator.train_on_batch(X_fake_and_real, y1)

      noise = tf.random.normal(shape=[batch_size, codings_size])
      y2 = tf.constant([[1.]] * batch_size)
      discriminator.trainable = False
      gan.train_on_batch(noise, y2)
      
</code>
</pre>

이 코드는 "핸즈온 머신러닝 2판"을 보고 공부할 때 연습했던 건데 원래 코드는 흑백 이미지(채널이 1개)에 맞춰져 있어서 컬러 이미지(채널이 3개)에도 적용할 수 있게 바꿨습니다.

참고로, 중간에 tf.cast(generated_images, tf.float64) 이 부분은 double tensor 에러가 뜨시는 분들을 위해 추가해놓은건데

**생성되는 이미지의 텐서 데이터 타입(코드에서는 generated_images를 말함)**과 **데이터 적재로 꺼내는 텐서 데이터 타입(코드에서는 X_batch를 말함)**이 다르면 double tensor에러로 충돌하게 됩니다.

따라서, 만약 에러가 뜨신다면 데이터 타입을 적절히 변환하셔야 합니다.

**noise** 는 generator에 넣어줄 랜덤 벡터를 말하는 겁니다.

쉽게 말해서, generator는 아무 값이나 입력받고 그걸 적절하게 소화시켜서 이미지를 뱉어낸다고 보시면 됩니다.

또, 먼저 판별자를 녹이고(trainable = True) 학습을 시켜준뒤 그 다음에 판별자를 다시 얼리고(trainable = False) gan을 학습시켜주는 것을 보실 수 있습니다.

이처럼 gan이 학습될때는 판별자의 가중치가 업데이트 되면 안됩니다. 오직 생성자만이 그 가중치를 전달받아 업데이트 되어야 하는 것 입니다.


그래서 이걸 실행시키면 다음과 같습니다.


<a id="2.4"></a>
## GAN 훈련

<pre>
<code>

train_gan(gan, dataset, batch_size, size, n_epochs=10)

</code>
</pre>

이때 저는 에폭을 10으로 설정했는데, 더 잘 훈련을 시키기 위해서는 에폭을 늘려야 합니다(혹은 하이퍼 파라미터 튜닝이 필요합니다).

그럼 10번의 에폭을 마치고나서 생성된 결과를 한번 볼게요.

<pre> 
<code>

plt.imshow(generator(noise)[0])

</code>
</pre>

[실행결과]

![image](https://user-images.githubusercontent.com/91790368/152838511-d29a1bf3-af4d-42e0-a0aa-3a9becd58648.png)

혹시 이게 뭔지 아시는 분...???


<a id="3"></a>
# Review



## 언제든 궁금하신 점이나 피드백은 환영입니다!

여기까지 읽어주셔서 감사합니다.

<a id="4"></a>
# References

Geron, A. (2019). Hands on machine learning with scikit-learn and tensorflow 2nd.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y. (2014). Generatvie Adversarial Nets, Advances in Neural Information Processing Systems 27 (NIPS 2014).
