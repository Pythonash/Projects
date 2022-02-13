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

그래서 오늘 할 것은 바로, YOLO 첫 번째 버젼인 **You only look once (YOLO)**의 간단한 논문 리뷰와 이러한 물체 탐지  구현해보기 입니다.

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
- [2. Code 구현 및 구조](#2)
    - [2.1 데이터셋](#2.1) 
    - [2.2 모델리뷰](#2.2)
    - [2.3 결과](#2.3)
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

![image](https://user-images.githubusercontent.com/91790368/153381866-a6f82795-f0d6-44e2-a5eb-5ea22b697ac9.png)

하나하나 구조를 뜯어가면서 설명하고 싶은데, 수식이 지금 안되는 터라...그냥 간략하게만 설명하면

첫 번째에는 좌표에 대한 loss, 두 번째는 넓이와 높이에 대한 loss인데 여기에 루트가 씌어져 있는 이유에 대해 논문에서는 다음과 같이 설명하고 있습니다.

**"오차제곱합은 큰 박스와 작은 박스의 에러를 같은 가중치를 두게 되는데, 큰 박스안에서의 작은 변화는 작은 박스안에서의 변화보다는 덜 문제가 된다.**

**이것을 부분적으로 해결하기 위해 넓이와 높이를 바로 계산하기 보다는 그것의 루트 제곱값을 계산한다.""**

여기서 주목하실 것은 격자셀 안에 **물체가 존재할 때에만 분류 오차에 규제를 가한다**는 것과

**가장 높은 IOU를 갖는 bounding box를 "responsible"한 예측기로 간주**하는데, 이것이 있을때에만 바운딩 박스 좌표에러에 패널티를 가한다는 것 입니다.

이후의 내용에는 64개의 배치, 0.9 momentum과 0.0005 decay 등 학습 파라미터에 대한 설명이 나오고, 학습률 스케쥴링을 에폭마다 어떻게 했는지, 또 과적합을 방지하기 위해 드롭아웃과 data augmentation을 진행 한 것 등등에 대한 설명이 나와 있습니다.


아무튼, 정리하자면

<a id="1.3"></a>
## YOLO 구조 정리

1. 물체탐지를 분류(classification)가 아닌 회귀(regression)로 접근한다.

2. 단일화된 네트워크로 구성되어 있고, 이미지에 대한 전체적인 정보를 학습한다.

3. 그렇기 때문에, 최적화가 간편하고 다른 물체들에 대한 적용 가능성도 높다.

---

<a id="2"></a>
## Code 구현 및 결과

모델 구현 과정이 너무 길어서 정확한 코드 및 진행 과정은 제 깃허브에 올려놓았습니다!

이 포스팅에서는 간단히 과정과 결과를 포스팅하겠습니다.

정확한 코드는 제 [깃허브](https://github.com/Pythonash/Projects)에 오셔서 **물체 탐지에 대한 노트북**을 클릭하시면 보다 상세하게 보실 수 있습니다.

<a id="2.1"></a>
## 데이터셋

YOLO의 저자들은 Pascal VOC dataset으로 검증을 했다고 합니다.

실제로 논문에 소개된 [홈페이지](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)에 들어가서 데이터셋을 찾으면 **Pascal VOC Dataset Mirror**를 다운 받으실 수 있습니다.

<img width="766" alt="스크린샷 2022-02-13 오후 11 30 01" src="https://user-images.githubusercontent.com/91790368/153757795-ef7ba875-a3e3-4398-b056-d4cb3fcde1eb.png">


되게 다크다크 하네요... 뭔가 더 전문적인 느낌이 듭니다. ㅋㅋㅋ

그럼 이제 데이터 셋을 다운 받으면 xml파일과 이미지 사진들이 포함되어 있는 tar 파일이 나옵니다.

이걸 압축해제하고 파싱하는 과정 또한 제 깃허브에 소개해놓았습니다.

이제 모델에 대한 간단한 리뷰를 하겠습니다.


<a id="2.2"></a>
## 모델

<pre>
<pre>Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 224, 224, 3  0           []                               
                                )]                                                                
                                                                                                  
 conv1_pad (ZeroPadding2D)      (None, 230, 230, 3)  0           ['input_1[0][0]']                
                                                                                                  
 conv1_conv (Conv2D)            (None, 112, 112, 64  9472        ['conv1_pad[0][0]']              
                                )                                                                 
                                                                                                  
 conv1_bn (BatchNormalization)  (None, 112, 112, 64  256         ['conv1_conv[0][0]']             
                                )                                                                 
                                                                                                  
 conv1_relu (Activation)        (None, 112, 112, 64  0           ['conv1_bn[0][0]']               
                                )                                                                 
                                                                                                  
 pool1_pad (ZeroPadding2D)      (None, 114, 114, 64  0           ['conv1_relu[0][0]']             
                                )                                                                 
                                                                                                  
 pool1_pool (MaxPooling2D)      (None, 56, 56, 64)   0           ['pool1_pad[0][0]']              
                                                                                                  
 conv2_block1_1_conv (Conv2D)   (None, 56, 56, 64)   4160        ['pool1_pool[0][0]']             
                                                                                                  
 conv2_block1_1_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block1_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block1_1_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block1_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block1_2_conv (Conv2D)   (None, 56, 56, 64)   36928       ['conv2_block1_1_relu[0][0]']    
                                                                                                  
 conv2_block1_2_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block1_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block1_2_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block1_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block1_0_conv (Conv2D)   (None, 56, 56, 256)  16640       ['pool1_pool[0][0]']             
                                                                                                  
 conv2_block1_3_conv (Conv2D)   (None, 56, 56, 256)  16640       ['conv2_block1_2_relu[0][0]']    
                                                                                                  
 conv2_block1_0_bn (BatchNormal  (None, 56, 56, 256)  1024       ['conv2_block1_0_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block1_3_bn (BatchNormal  (None, 56, 56, 256)  1024       ['conv2_block1_3_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block1_add (Add)         (None, 56, 56, 256)  0           ['conv2_block1_0_bn[0][0]',      
                                                                  'conv2_block1_3_bn[0][0]']      
                                                                                                  
 conv2_block1_out (Activation)  (None, 56, 56, 256)  0           ['conv2_block1_add[0][0]']       
                                                                                                  
 conv2_block2_1_conv (Conv2D)   (None, 56, 56, 64)   16448       ['conv2_block1_out[0][0]']       
                                                                                                  
 conv2_block2_1_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block2_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block2_1_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block2_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block2_2_conv (Conv2D)   (None, 56, 56, 64)   36928       ['conv2_block2_1_relu[0][0]']    
                                                                                                  
 conv2_block2_2_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block2_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block2_2_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block2_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block2_3_conv (Conv2D)   (None, 56, 56, 256)  16640       ['conv2_block2_2_relu[0][0]']    
                                                                                                  
 conv2_block2_3_bn (BatchNormal  (None, 56, 56, 256)  1024       ['conv2_block2_3_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block2_add (Add)         (None, 56, 56, 256)  0           ['conv2_block1_out[0][0]',       
                                                                  'conv2_block2_3_bn[0][0]']      
                                                                                                  
 conv2_block2_out (Activation)  (None, 56, 56, 256)  0           ['conv2_block2_add[0][0]']       
                                                                                                  
 conv2_block3_1_conv (Conv2D)   (None, 56, 56, 64)   16448       ['conv2_block2_out[0][0]']       
                                                                                                  
 conv2_block3_1_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block3_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block3_1_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block3_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block3_2_conv (Conv2D)   (None, 56, 56, 64)   36928       ['conv2_block3_1_relu[0][0]']    
                                                                                                  
 conv2_block3_2_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block3_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block3_2_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block3_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block3_3_conv (Conv2D)   (None, 56, 56, 256)  16640       ['conv2_block3_2_relu[0][0]']    
                                                                                                  
 conv2_block3_3_bn (BatchNormal  (None, 56, 56, 256)  1024       ['conv2_block3_3_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block3_add (Add)         (None, 56, 56, 256)  0           ['conv2_block2_out[0][0]',       
                                                                  'conv2_block3_3_bn[0][0]']      
                                                                                                  
 conv2_block3_out (Activation)  (None, 56, 56, 256)  0           ['conv2_block3_add[0][0]']       
                                                                                                  
 conv3_block1_1_conv (Conv2D)   (None, 28, 28, 128)  32896       ['conv2_block3_out[0][0]']       
                                                                                                  
 conv3_block1_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block1_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block1_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block1_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block1_2_conv (Conv2D)   (None, 28, 28, 128)  147584      ['conv3_block1_1_relu[0][0]']    
                                                                                                  
 conv3_block1_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block1_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block1_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block1_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block1_0_conv (Conv2D)   (None, 28, 28, 512)  131584      ['conv2_block3_out[0][0]']       
                                                                                                  
 conv3_block1_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block1_2_relu[0][0]']    
                                                                                                  
 conv3_block1_0_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block1_0_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block1_3_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block1_3_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block1_add (Add)         (None, 28, 28, 512)  0           ['conv3_block1_0_bn[0][0]',      
                                                                  'conv3_block1_3_bn[0][0]']      
                                                                                                  
 conv3_block1_out (Activation)  (None, 28, 28, 512)  0           ['conv3_block1_add[0][0]']       
                                                                                                  
 conv3_block2_1_conv (Conv2D)   (None, 28, 28, 128)  65664       ['conv3_block1_out[0][0]']       
                                                                                                  
 conv3_block2_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block2_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block2_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block2_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block2_2_conv (Conv2D)   (None, 28, 28, 128)  147584      ['conv3_block2_1_relu[0][0]']    
                                                                                                  
 conv3_block2_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block2_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block2_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block2_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block2_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block2_2_relu[0][0]']    
                                                                                                  
 conv3_block2_3_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block2_3_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block2_add (Add)         (None, 28, 28, 512)  0           ['conv3_block1_out[0][0]',       
                                                                  'conv3_block2_3_bn[0][0]']      
                                                                                                  
 conv3_block2_out (Activation)  (None, 28, 28, 512)  0           ['conv3_block2_add[0][0]']       
                                                                                                  
 conv3_block3_1_conv (Conv2D)   (None, 28, 28, 128)  65664       ['conv3_block2_out[0][0]']       
                                                                                                  
 conv3_block3_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block3_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block3_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block3_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block3_2_conv (Conv2D)   (None, 28, 28, 128)  147584      ['conv3_block3_1_relu[0][0]']    
                                                                                                  
 conv3_block3_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block3_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block3_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block3_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block3_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block3_2_relu[0][0]']    
                                                                                                  
 conv3_block3_3_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block3_3_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block3_add (Add)         (None, 28, 28, 512)  0           ['conv3_block2_out[0][0]',       
                                                                  'conv3_block3_3_bn[0][0]']      
                                                                                                  
 conv3_block3_out (Activation)  (None, 28, 28, 512)  0           ['conv3_block3_add[0][0]']       
                                                                                                  
 conv3_block4_1_conv (Conv2D)   (None, 28, 28, 128)  65664       ['conv3_block3_out[0][0]']       
                                                                                                  
 conv3_block4_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block4_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block4_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block4_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block4_2_conv (Conv2D)   (None, 28, 28, 128)  147584      ['conv3_block4_1_relu[0][0]']    
                                                                                                  
 conv3_block4_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block4_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block4_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block4_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block4_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block4_2_relu[0][0]']    
                                                                                                  
 conv3_block4_3_bn (BatchNormal  (None, 28, 28, 512)  2048       ['conv3_block4_3_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block4_add (Add)         (None, 28, 28, 512)  0           ['conv3_block3_out[0][0]',       
                                                                  'conv3_block4_3_bn[0][0]']      
                                                                                                  
 conv3_block4_out (Activation)  (None, 28, 28, 512)  0           ['conv3_block4_add[0][0]']       
                                                                                                  
 conv4_block1_1_conv (Conv2D)   (None, 14, 14, 256)  131328      ['conv3_block4_out[0][0]']       
                                                                                                  
 conv4_block1_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block1_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block1_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block1_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block1_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block1_1_relu[0][0]']    
                                                                                                  
 conv4_block1_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block1_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block1_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block1_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block1_0_conv (Conv2D)   (None, 14, 14, 1024  525312      ['conv3_block4_out[0][0]']       
                                )                                                                 
                                                                                                  
 conv4_block1_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block1_2_relu[0][0]']    
                                )                                                                 
                                                                                                  
 conv4_block1_0_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block1_0_conv[0][0]']    
 ization)                       )                                                                 
                                                                                                  
 conv4_block1_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block1_3_conv[0][0]']    
 ization)                       )                                                                 
                                                                                                  
 conv4_block1_add (Add)         (None, 14, 14, 1024  0           ['conv4_block1_0_bn[0][0]',      
                                )                                 'conv4_block1_3_bn[0][0]']      
                                                                                                  
 conv4_block1_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block1_add[0][0]']       
                                )                                                                 
                                                                                                  
 conv4_block2_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block1_out[0][0]']       
                                                                                                  
 conv4_block2_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block2_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block2_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block2_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block2_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block2_1_relu[0][0]']    
                                                                                                  
 conv4_block2_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block2_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block2_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block2_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block2_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block2_2_relu[0][0]']    
                                )                                                                 
                                                                                                  
 conv4_block2_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block2_3_conv[0][0]']    
 ization)                       )                                                                 
                                                                                                  
 conv4_block2_add (Add)         (None, 14, 14, 1024  0           ['conv4_block1_out[0][0]',       
                                )                                 'conv4_block2_3_bn[0][0]']      
                                                                                                  
 conv4_block2_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block2_add[0][0]']       
                                )                                                                 
                                                                                                  
 conv4_block3_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block2_out[0][0]']       
                                                                                                  
 conv4_block3_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block3_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block3_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block3_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block3_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block3_1_relu[0][0]']    
                                                                                                  
 conv4_block3_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block3_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block3_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block3_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block3_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block3_2_relu[0][0]']    
                                )                                                                 
                                                                                                  
 conv4_block3_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block3_3_conv[0][0]']    
 ization)                       )                                                                 
                                                                                                  
 conv4_block3_add (Add)         (None, 14, 14, 1024  0           ['conv4_block2_out[0][0]',       
                                )                                 'conv4_block3_3_bn[0][0]']      
                                                                                                  
 conv4_block3_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block3_add[0][0]']       
                                )                                                                 
                                                                                                  
 conv4_block4_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block3_out[0][0]']       
                                                                                                  
 conv4_block4_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block4_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block4_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block4_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block4_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block4_1_relu[0][0]']    
                                                                                                  
 conv4_block4_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block4_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block4_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block4_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block4_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block4_2_relu[0][0]']    
                                )                                                                 
                                                                                                  
 conv4_block4_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block4_3_conv[0][0]']    
 ization)                       )                                                                 
                                                                                                  
 conv4_block4_add (Add)         (None, 14, 14, 1024  0           ['conv4_block3_out[0][0]',       
                                )                                 'conv4_block4_3_bn[0][0]']      
                                                                                                  
 conv4_block4_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block4_add[0][0]']       
                                )                                                                 
                                                                                                  
 conv4_block5_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block4_out[0][0]']       
                                                                                                  
 conv4_block5_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block5_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block5_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block5_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block5_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block5_1_relu[0][0]']    
                                                                                                  
 conv4_block5_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block5_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block5_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block5_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block5_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block5_2_relu[0][0]']    
                                )                                                                 
                                                                                                  
 conv4_block5_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block5_3_conv[0][0]']    
 ization)                       )                                                                 
                                                                                                  
 conv4_block5_add (Add)         (None, 14, 14, 1024  0           ['conv4_block4_out[0][0]',       
                                )                                 'conv4_block5_3_bn[0][0]']      
                                                                                                  
 conv4_block5_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block5_add[0][0]']       
                                )                                                                 
                                                                                                  
 conv4_block6_1_conv (Conv2D)   (None, 14, 14, 256)  262400      ['conv4_block5_out[0][0]']       
                                                                                                  
 conv4_block6_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block6_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block6_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block6_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block6_2_conv (Conv2D)   (None, 14, 14, 256)  590080      ['conv4_block6_1_relu[0][0]']    
                                                                                                  
 conv4_block6_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block6_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block6_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block6_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block6_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block6_2_relu[0][0]']    
                                )                                                                 
                                                                                                  
 conv4_block6_3_bn (BatchNormal  (None, 14, 14, 1024  4096       ['conv4_block6_3_conv[0][0]']    
 ization)                       )                                                                 
                                                                                                  
 conv4_block6_add (Add)         (None, 14, 14, 1024  0           ['conv4_block5_out[0][0]',       
                                )                                 'conv4_block6_3_bn[0][0]']      
                                                                                                  
 conv4_block6_out (Activation)  (None, 14, 14, 1024  0           ['conv4_block6_add[0][0]']       
                                )                                                                 
                                                                                                  
 conv5_block1_1_conv (Conv2D)   (None, 7, 7, 512)    524800      ['conv4_block6_out[0][0]']       
                                                                                                  
 conv5_block1_1_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block1_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block1_1_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block1_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block1_2_conv (Conv2D)   (None, 7, 7, 512)    2359808     ['conv5_block1_1_relu[0][0]']    
                                                                                                  
 conv5_block1_2_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block1_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block1_2_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block1_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block1_0_conv (Conv2D)   (None, 7, 7, 2048)   2099200     ['conv4_block6_out[0][0]']       
                                                                                                  
 conv5_block1_3_conv (Conv2D)   (None, 7, 7, 2048)   1050624     ['conv5_block1_2_relu[0][0]']    
                                                                                                  
 conv5_block1_0_bn (BatchNormal  (None, 7, 7, 2048)  8192        ['conv5_block1_0_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block1_3_bn (BatchNormal  (None, 7, 7, 2048)  8192        ['conv5_block1_3_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block1_add (Add)         (None, 7, 7, 2048)   0           ['conv5_block1_0_bn[0][0]',      
                                                                  'conv5_block1_3_bn[0][0]']      
                                                                                                  
 conv5_block1_out (Activation)  (None, 7, 7, 2048)   0           ['conv5_block1_add[0][0]']       
                                                                                                  
 conv5_block2_1_conv (Conv2D)   (None, 7, 7, 512)    1049088     ['conv5_block1_out[0][0]']       
                                                                                                  
 conv5_block2_1_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block2_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block2_1_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block2_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block2_2_conv (Conv2D)   (None, 7, 7, 512)    2359808     ['conv5_block2_1_relu[0][0]']    
                                                                                                  
 conv5_block2_2_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block2_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block2_2_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block2_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block2_3_conv (Conv2D)   (None, 7, 7, 2048)   1050624     ['conv5_block2_2_relu[0][0]']    
                                                                                                  
 conv5_block2_3_bn (BatchNormal  (None, 7, 7, 2048)  8192        ['conv5_block2_3_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block2_add (Add)         (None, 7, 7, 2048)   0           ['conv5_block1_out[0][0]',       
                                                                  'conv5_block2_3_bn[0][0]']      
                                                                                                  
 conv5_block2_out (Activation)  (None, 7, 7, 2048)   0           ['conv5_block2_add[0][0]']       
                                                                                                  
 conv5_block3_1_conv (Conv2D)   (None, 7, 7, 512)    1049088     ['conv5_block2_out[0][0]']       
                                                                                                  
 conv5_block3_1_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block3_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block3_1_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block3_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block3_2_conv (Conv2D)   (None, 7, 7, 512)    2359808     ['conv5_block3_1_relu[0][0]']    
                                                                                                  
 conv5_block3_2_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block3_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block3_2_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block3_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block3_3_conv (Conv2D)   (None, 7, 7, 2048)   1050624     ['conv5_block3_2_relu[0][0]']    
                                                                                                  
 conv5_block3_3_bn (BatchNormal  (None, 7, 7, 2048)  8192        ['conv5_block3_3_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block3_add (Add)         (None, 7, 7, 2048)   0           ['conv5_block2_out[0][0]',       
                                                                  'conv5_block3_3_bn[0][0]']      
                                                                                                  
 conv5_block3_out (Activation)  (None, 7, 7, 2048)   0           ['conv5_block3_add[0][0]']       
                                                                                                  
 global_average_pooling2d (Glob  (None, 2048)        0           ['conv5_block3_out[0][0]']       
 alAveragePooling2D)                                                                              
                                                                                                  
 flatten (Flatten)              (None, 2048)         0           ['global_average_pooling2d[0][0]'
                                                                 ]                                
                                                                                                  
 dropout (Dropout)              (None, 2048)         0           ['flatten[0][0]']                
                                                                                                  
 batch_normalization (BatchNorm  (None, 2048)        8192        ['dropout[0][0]']                
 alization)                                                                                       
                                                                                                  
 dense (Dense)                  (None, 1200)         2458800     ['batch_normalization[0][0]']    
                                                                                                  
 leaky_re_lu (LeakyReLU)        (None, 1200)         0           ['dense[0][0]']                  
                                                                                                  
 dropout_1 (Dropout)            (None, 1200)         0           ['leaky_re_lu[0][0]']            
                                                                                                  
 batch_normalization_1 (BatchNo  (None, 1200)        4800        ['dropout_1[0][0]']              
 rmalization)                                                                                     
                                                                                                  
 dense_1 (Dense)                (None, 1000)         1201000     ['batch_normalization_1[0][0]']  
                                                                                                  
 leaky_re_lu_1 (LeakyReLU)      (None, 1000)         0           ['dense_1[0][0]']                
                                                                                                  
 dense_2 (Dense)                (None, 1)            1001        ['leaky_re_lu_1[0][0]']          
                                                                                                  
 dense_3 (Dense)                (None, 1)            1001        ['leaky_re_lu_1[0][0]']          
                                                                                                  
 dense_4 (Dense)                (None, 1)            1001        ['leaky_re_lu_1[0][0]']          
                                                                                                  
 dense_5 (Dense)                (None, 1)            1001        ['leaky_re_lu_1[0][0]']          
                                                                                                  
 dense_6 (Dense)                (None, 1)            1001        ['leaky_re_lu_1[0][0]']          
                                                                                                  
==================================================================================================
Total params: 27,265,509
Trainable params: 3,671,301
Non-trainable params: 23,594,208
__________________________________________________________________________________________________
</pre>
</pre>

사실 물체 탐지와 같은 모델은 단순한 모델이 아니기 때문에 모델을 구축하고 훈련시키는 등의 과정이 일반적인 모델보다 딥합니다. 

예측해야할 타겟이 어떤 하나의 값이 아니라 사진을 인식하고, 그 사진에서도 바운딩 박스에 대한 좌표를 예측해야하기 때문이죠.

실제로도 논문에서는 7x7x30의 tensor를 아웃풋으로 내기도 하는데 훈련에도 굉장히 오래 걸렸다고 합니다.

하지만 이번에 할 것은 단순히 **구현**정도라 전이 학습(transfer learning)을 이용하겠습니다.

위에 요약된 모델은 ResNet50이라는 사전 학습된 모델을 불러오고, 그 위에 아웃풋 구조를 물체 탐지에 맞게 다시 재조정 해놓은 모델 입니다.

논문 저자들은 **Darknet**이라는 모델을 이용했다고 하는데 이 부분은 나중에 시간이 되면 포스팅 해보겠습니다.

그럼 빠르게 결과를 보도록 하겠습니다.

<a id="2.3"></a>
## 결과

복잡한 데이터 전처리와 모델을 구현하고 훈련을 한뒤, 제 이미지를 넣어주면 다음과 같습니다.

<pre>
<code>
result = detection.predict(np.array([input_img]))
ss = cv2.imread('./imagee.jpg')
a = result[0][0][0] *(ss.shape[0]/224)
b = result[1][0][0] *(ss.shape[1]/224)
c = result[2][0][0] *(ss.shape[0]/224)
d = result[3][0][0] *(ss.shape[1]/224)
img_1 = Image.open('imagee.jpg').convert("RGB")
draw = ImageDraw.Draw(img_1)
draw.rectangle(((a, b), (c, d)), outline="red")
draw.text((a, b), 'person')
plt.figure(figsize=(25,10))
plt.imshow(img_1)
plt.show()
plt.close()
</code>
</pre>

[실행결과]
![image](https://user-images.githubusercontent.com/91790368/153758416-91592004-4471-4e79-b436-f5a657d66f8d.png)

신기하네요! 

모델을 충분히 훈련하지 않아서 바운딩 박스의 위치에 오차가 있긴 하지만

제 사진을 인식하고 바운딩 박스로 저의 위치를 표시해 줍니다.

이렇게 한번 물체 탐지 알고리즘 구현을 해보았습니다.

<a id="3"></a>
# Review

사실 딥러닝으로 할 수 있는게 정말 무궁무진해서 시간만 많다면 더 정교하고 신기한 모델을 구현해보고 싶었는데, 시간적으로도 그렇고 요즘 논문쓰기 바빠서 단순 구현만 해보았습니다 ㅎㅎ

YOLO나 물체 탐지 알고리즘에 대해서는 다른 블로그나 자료에도 이론적으로 상세하게 포스팅된 것들이 많아서 내용위주 보다는 최대한 도움이 되게끔 코드로 구현해보았습니다.

더군다나 요즘은 pytorch보다는 tensorflow와 keras가 더 많이 이용되는 것 같아서 쉽게 따라하실 수 있도록 구현을 해보았는데요, 도움이 되었으면 좋겠습니다.

이번 알고리즘을 구현하면서 까먹었던 부분이나 데이터를 처리하고 핸들링하는 스킬에 대해서도 많은 공부가 되어서 더 좋았습니다.

## 언제든 궁금하신 점이나 피드백은 환영입니다!

여기까지 읽어주셔서 감사합니다.

<a id="4"></a>
# References

1. Redmon, J., Divvala, S., Girshick, R., Farhadi, A. (2016). You only look once: Unified, real-Time object detection. CVPR, 779-788.

2. https://bkshin.tistory.com/entry/OpenCV-6-dd - 이미지에 바운딩 박스 그리는 데에 참고

3. https://lee-mandu.tistory.com/519?category=838684 - 데이터 파일 파싱하는 데에 
