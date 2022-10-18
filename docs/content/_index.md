## Summary

Sign language is a visual communication skill that enables individuals with different types of hearing impairment to communicate in society. It is the language used by most deaf people in daily lives and, moreover, the symbol of identification between the members of that community and the main force that unites them.

Research related to sign languages has been developed since the 1990s and the main challenges are primarily related to considering the dynamic aspects of the language, such as movements, articulations between body parts and non-manual expressions, rather than merely recognizing static signs or isolated hand positions.

![sign language](img/sl.png)

The related work on ASL has received huge amounts of progress throughout the recent years, not only the development of the sensors but also techniques on image feature extraction have been widely used to improve the preprocessing outcomes of sign languages.

Besides that, Convolutional neural networks also make the recognition accuracy up to 90%, depending on specific dataset. Variations of CNN such as 3D-CNN and Recurrent Neural Network also help achieve interesting results. However, the challenge still lies in the  dynamics of languages such as its movements, non-manual expressions, and articulations between parts of the body. In this sense, it is extremely relevant that new studies observe such important characteristics.

The datasets our group chose is Phoenix-14t: Parallel Corpus of Sign language Videos, Gloss and Translation [[2]]({{< relref path="_index.md#ref2">}}). This dataset is extracted from weather forecast airings of the German tv station PHOENIX. Interpreters wear dark clothes in front of an artificial gray background with color transition. All recorded videos are at 25 frames per second and the size of the frames is 210 by 260 pixels. Each frame shows the interpreter box only. The dataset consists of a parallel corpus of German sign language videos from 9 different signers, gloss-level annotations with a vocabulary of 1,066 different signs and translations into German spoken language with a vocabulary of 2,887 different words. There are 7,096 training pairs, 519 development and 642 test pairs.


## Research Plan
Our project can be divided into two parts, the first is extracting sign language glosses from videos and the second is translating spoken language from the sign language glosses. Considering the feasibility and complexity, we'll try to implement the model proposed in Better Sign Language Translation with STMC-Transformer [[1]]({{< relref path="_index.md#ref1">}}) first. The authors used a Spatial-Temporal Multi-Cue (STMC) Network to efficiently process multiple visual cues (face, hand, full-frame, and pose) from sign language video. Then a two-layer Transformer was constructed to translate extracted sign language glosses.
 
![model architecture](img/architecture.png "Figure: STMC-Transformer network for SLT. PE: Positional Encoding, MHA: Multihead Attention, FF: Feed Forward.")

PHOENIX-Weather 2014T [[2]]({{< relref path="_index.md#ref2">}}) is one of the mostly used datasets in SLR and we choose it as our dataset for easier model comparison. Evaluation will be based on two metrics, WER (Word Error Rate) for sign language recognition and BLEU (bilingual evaluation understudy) for translating sign language glosses to spoken language. WER is the ratio of errors in glosses to the total expressed words and BLEU score is a number between zero and one that measures the similarity of the machine-translated text to a set of high quality reference translations. 

To extend further, we'll try some SOTA models of each part, e.g., SMKD[[3]]({{< relref path="_index.md#ref3">}}) for sign language recognition, to see if the whole model can achieve a better performance. At the same time, considering there are some other sign language datasets like American Sign Language Lexicon Video Dataset (ASLLVD) [[4]]({{< relref path="_index.md#ref4">}}), we'll try to use techniques like transfer learning to apply our model to other sign languages.

## Feasibility
The project undertaken is a full semester project as it would be done in two parts. The first part is recognition and the second part is translation. By the first progress updates, we aim to complete the starting part of data importing and pre-processing. Converting the videos into a format that can be used as an input for our models. And in the following weeks after that, we will simultaneously try to develop the model specified above and run them on the data we have and evaluate our results. Once we have decent and satisfactory results, we will move onto the next part which is the translation. We should be able to start the translation part by the second week of November. If necessary, we will do some data preprocessing again on the results obtained and then develop the transformer model as mentioned in the research plan. Following that, once we have the results, we will also apply the SOTA models and check if we can achieve better accuracy. In the final two weeks, we will work on various transfer learning techniques to apply our model on the other sign languages and also work on the report and final presentation simultaneously.

The risks of this project is the computational power we need for running our models on the videos. The other risk is not being able to achieve decent accuracy for the recognition part as some of the hand signs can look similar which could confuse the model. This could make the translation task more difficult. Also, as this dataset is restricted to the weather forecast domain, we may not be able to fully assess the performance of our model in a more general domain. To overcome the final risk, we will therefore work on transfer learning techniques to evaluate the model on other sign languages too. 

As a criteria for success, we will use the metrics for BLEU and WER as described in the research plan. Taking the model proposed in [[1]]({{< relref path="_index.md#ref1">}}) as the baseline, the WER of our final model should be lower than 21.0 and BLEU-4 should be higher than 21.65.

## References

##### [1] Yin, Kayo, and Jesse Read. "Better sign language translation with STMC-transformer." Proceedings of the 28th International Conference on Computational Linguistics. 2020. {#ref1}

##### [2] Camgoz, Necati Cihan, et al. "Neural sign language translation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018. {#ref2}

##### [3] Hao, Aiming, Yuecong Min, and Xilin Chen. "Self-Mutual Distillation Learning for Continuous Sign Language Recognition." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. {#ref3}

##### [4] Athitsos, Vassilis, et al. "The american sign language lexicon video dataset." 2008 IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops. IEEE, 2008. {#ref4}

