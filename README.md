## PU-learning
1. Convex Formulation of Multiple Instance Learning from Positive and Unlabeled Bags
   
    key: pu | mil | ml

    [paper](https://arxiv.org/pdf/1704.06767.pdf) May 2018 

    notes: 09/09/2019

    >This paper use a special kernel, set kernel, to deal with multi-instance learning from a bag level. The model is named PU-SKC, which can be seen as a SVM based model. The method can be also extended to instance-level classification.

    related paper (not carefully readed):

    a. Semi-Supervised AUC Optimization based on
    Positive-Unlabeled Learning [paper](https://arxiv.org/pdf/1705.01708.pdf)


## Seq2Seq

1. **Synchronous Bidirectional Neural Machine Translation** 

    key: Seq2Seq | bidirectional decoding

    [paper](https://arxiv.org/pdf/1905.04847.pdf)
    [blog chinese](https://spaces.ac.cn/archives/6877) May 2019 

    notes: 09/10/2019

    >The conventional Seq2Seq model have high precision at the beginning of the decoding but it decreases as the time step prolonged. To solve this problem, this paper propose a novel bidirectional decoding for Seq2Seq. 

    > Concern: while the performance of the first time-steps and last time-steps increase, the accuracy in the middle might be still low.
    > Moreover, Info leak? 

## Heng Ji

1. **Cross-lingual Structure Transfer for Relation and Event Extraction**
   
   key: xlie | relation and event extraction | GCN

   [paper](http://nlp.cs.rpi.edu/paper/crosslingualstructure2019.pdf)

   notes: 09/11/2019

   > This paper leverage cross-lingual features, both symbolic and distributional, for extracting universal common features from the text. The results show that these features are well-transferable and achieve comparable performance with the monolingual model.


2. **Cross-lingual Multi-Level Adversarial Transfer to Enhance Low-Resource Name Tagging**

    key: xlie | ner | gan

    [paper](http://nlp.cs.rpi.edu/paper/adversarial2019.pdf)

3. **Multilingual Entity, Relation, Event and Human Value Extraction**
   
    key: xlie | multi-task

    [paper](http://nlp.cs.rpi.edu/paper/naacldemo2019.pdf)


4. **Low-Resource Name Tagging Learned with Weakly Labeled Data**

    key: xlie | ner | weak supervised

    [paper](http://nlp.cs.rpi.edu/paper/weaklysupervised2019.pdf)

    notes: 09/12/2019

    > This paper propose a novel Partial CRF with O entity sampling trick. The conventional PCRF cannot learn O entity. By simple rule, reliable O samples can be found, thus enhancing the overall performance in weak supervised setting.

    > simply drop the wrong labeled data is bad. 
    
    > "However, abandoning training data may exacerbate the issue of inadequate annotation. Therefore, maximizing the potential of massive noisy data as well as high quality part, yet being efficient, is challenging."

## Metric Learning

1. **SoftTriple Loss: Deep Metric Learning Without Triplet Sampling**
   
   key: sampling | triplet loss | SoftTriple Loss

   [paper](https://arxiv.org/pdf/1909.05235.pdf)

   notes: 09/13/2019

   1. This paper introduce multiple centers for each class. 
   2. Training without sampling. Triple numbers are linear in the examples (rather than O(n^3)). This is similar to that of ProtoNet.

## Multi-Label

1. **Learning to Learn and Predict: A Meta-Learning Approach for Multi-Label Classification**

    [paper](https://arxiv.org/pdf/1909.04176.pdf)

    key: meta-learning | ? label dependency | multi-label classification

    notes: 09/13/2019

    > This paper propose a RNN-based meta-learner to learn the the threshold and instance weight for loss, that is, a variant of standard cross-entropy.



## back-translation

1. **Neural Machine Translation of Low-Resource and Similar Languages with Backtranslation**

    [paper](https://www.aclweb.org/anthology/W19-5431)

    key: NMT | back-translation | data augmentation

    notes: 09/13/2019

    > back-translation for data augmentation

    > hyperparameter tuning in WMT task

    > noisy synthesised data is better than beam-search or greedy search.

2.  **Understanding Back-Translation at Scale**

    key: NMT | back-translation | data augmentation

    [paper](https://aclweb.org/anthology/D18-1045)

    notes: 09/13/2019

    > noisy synthesised data is better than beam-search or greedy search.

## class-imbalanced classification

1. **Imbalance Problems in Object Detection: A Review**

    [zhihu](https://zhuanlan.zhihu.com/p/82371629)

    notes: 09/13/2019

    key: class imbalance | object detection | review

    > foreground & background classification ?

    > a. [Imbalance Problems in Object Detection: A Review](https://arxiv.org/pdf/1909.00169.pdf)
    b. [paper list](https://github.com/kemaloksuz/ObjectDetectionImbalance)


## Cross-Lingual
1. **Entity Projection via Machine-Translation for Cross-Lingual NER** EMNLP2019
   
   [paper](https://arxiv.org/pdf/1909.05356.pdf)

   notes: 09/15/2019

   key: xlie | alignment | NER

   > This paper use complicated rule for projecting entity tags by leveraging off-the-shelf MT system.

2. **A Discriminative Neural Model for Cross-Lingual Word Alignment**
   
   [paper](https://arxiv.org/pdf/1909.00444.pdf)

   key: xlie | alignment | NER

   notes: 09/20/2019

   > This paper can be seen as a kind of multi-head selection. That is, the Cartesian Product of source x target (mutli-head is source x source)

   > It mention a very important trick! "alignments are context sensitive": the decision to align the current position will affect to its neighbors. So they add a 3x3 conv layer! This layer brings about 25 F1 score improvement. 

3. **Neural Network Alignment for Sentential Paraphrases**

    [paper](https://www.aclweb.org/anthology/P19-1467)


4. **On The Alignment Problem In Multi-Head Attention-Based Neural Machine Translation**

    [paper](https://www.aclweb.org/anthology/W18-6318)

    key: NMT | alignment

    notes: 09/20/2019

    > gold annotation + hard constraint

5. **A Little Annotation does a Lot of Good: A Study in Bootstrapping Low-resource Named Entity Recognizers**

    [paper](https://arxiv.org/pdf/1908.08983.pdf)

## Machine Translation
1. **Levenshtein Transformer**
   
    [paper](https://arxiv.org/pdf/1905.11006.pdf)

    notes: 09/15/2019

    key: machine translation | alignment | generation

    > This paper use the action sequence of edit distance to supervise sentence editing. action space: insertion and deletion.

    > problem: The insertion and deletion might never end because neural network is a blackbox.

