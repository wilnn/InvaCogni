# InvaCogni: A Domain Invariant Multimodal and Multilingual Classifier for Mild Cognitive Impairment
<!-- table of contents-->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Introduction">Introduction</a>
    </li>
    <li><a href="#Method">Method</a><ol>
      <li><a href="#Gradient-Reversal-Layer-(GRL)">Gradient Reversal Layer (GRL)</a></li>
      <li><a href="#Training">Training</a><ol>
        <li><a href="#setting-1:-The-InvaCogni-model-without-GRL">setting 1: The InvaCogni model without GRL</a></li>
      </ol></li>
    </ol></li>
    <li><a href="#How-It-Works">How It Works</a></li>
    <li><a href="#Example/Demonstration">Examples/Demonstration</a></li>
    <li><a href="#Issue">Issue</a></li>
    <li><a href="#Possible-Improvements">Possible Improvements</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## Introduction:
- Mild Cognitive Impairment (MCI) is a condition in which the patient has noticeable declines in cognitive abilities, like memory, attention, or thinking skills, that are greater than expected for a person’s age but not severe enough to interfere significantly with daily life or independent function. People with MCI might notice that they forget things more often, have trouble finding words, or struggle with complex tasks, but they can generally still live independently [5]. They also have a high risk of developing dementia, which is the condition when a person has a loss of memory, language, problem-solving, and other thinking abilities that are severe enough to interfere with daily life [6]. Diagnosing whether a patient has MCI or not can be a crucial step to prevent it from developing into dementia, which currently has no cure and affects millions of people worldwide. To determine if someone has MCI or not, we can evaluate their cognitive ability; the idea here is that people with MCI will have a harder time describing, recognizing, and finding the word to describe the images and events.
- There has been some research on creating a classifier for this task using the Taukdial dataset from DementiaBank. One research developed a model called CogniVoice, which is a multimodal and multilingual network that uses Product of Experts (PoEs) to mitigate reliance on shortcut solutions (the easy-to-find patterns or spurious correlations in the training data). This model has the Unweighted Average Recall (UAR) score, which varies a lot across patients from different groups: 11.8% difference between male and female and 19.1% difference between English and Chinese [2]. This can indicate that the model is using domain-specific information (gender, accents, languages, etc.) when making predictions instead of the more general information that is related to the task and is very likely just a correlation and does not have any causal relationship with the MCI problem.
- In this project, I will create a multimodal and multilingual classifier for classifying whether a person has MCI based on their ability to describe and recognize events which directly measure their cognitive ability. Additionally, I will use a method which is called Gradient Reversal Layer (GRL) [3], to reduce the domain differences and by that, reduce the model performance gap on different domains (between male and female, English and Chinese).
The base version of InvaCogni (without GRL) outperforms baseline models by over 1% on F1 score and by over 10% on Unweighted Average Recall (UAR). When GRL is applied, overall performance drops by about 4%, but the performance gap between male and female groups is reduced to only a 2% difference for UAR and a 0.6% difference for F1, and the model’s performance gap between English and Chinese speakers reduces to a 7.1% difference for the UAR score.

## Method
### Gradient Reversal Layer (GRL)
Gradient Reversal Layer (GRL) is a technique introduced in [3] that can reduce the differences between domains by making the data representations output by the encoder for different domains become indistinguishable in terms of those domains. By using this, I can effectively reduce the domain differences in the data, which will reduce the performance gap between those domains.
<img alt="image" src="./assets/grl_forwardpass.png" />
<br>Figure 1. showing how GRL works during the forward pass for male and female domains.
<br><br><img alt="image" src="./assets/grl_backprop.png" />
<br>Figure 2. showing how GRL works during the backward pass. The parameters in the domain classifier are trying to minimize the loss, while the parameters in the audio encoder are trying to maximize the loss due to the GRL.<br>
<br>In Figure 2, this setup creates a situation where the encoder and the domain classifier compete. The domain classifier is trying to update its parameters in a way that can help it tell which input is which domain (male or female), while the audio encoder will be based on that to update its parameters in the way that makes its output harder for the domain classifier to tell which input is which domain (meaning it tries to make the data representations output from the audio encoder to be viewed as the same in terms of the domains, which are male and female in this case).
<br>In figure 2, $$λ_GRL$$ is -1 for all cases, and λ_loss  is a number between 0 and 1 that will be adjusted during training based on this scheduler from [3]:
<br><img height="150px" width="300px"  alt="image" src="./assets/lambda_loss_scheduler.png" />        (1)
<br>Here, the $$\gamma$$ is a constant that controls how fast the $$\lambda_{loss}$$ will increase, and $$p$$ is the ratio of $$\frac {\text{the current training step}}{\text{total training steps}}$$ 
### Training:
#### setting 1: The InvaCogni model without GRL
<img alt="image" src="./assets/invacogni_no_grl.png" />
<br>Figure 3. The architecture of the InvaCogni model with no GRL. 
<br><br>I use the image encoder of siglip-base [7] as the image encoder of this model. For the text encoder, I use the BERT-base-multilingual model [4], and for the audio encoder, I use the encoder part of the whisper-base model [1]. The embeddings from the image encoder and text encoder will be fused using cross attention. The idea here is that, given the patients’ text description of the image, the model should learn to match/compare the text description with the corresponding image. Then, the output from the transformer block will be concatenated with features extracted from the audio embeddings and passed to the task classifier to give the prediction.
<br>The text encoder and image encoder will be frozen during training.
