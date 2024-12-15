# Sentiment Analysis of Sports-Related Content Using Fine-Tuned BERT Model: Unveiling Public Reactions to Sports Events

[![CI Run](https://github.com/Lulu-1121/Sports-Sentiment-BERT/actions/workflows/ci.yml/badge.svg)](https://github.com/Lulu-1121/Sports-Sentiment-BERT/actions/workflows/ci.yml)

## **Overview**
This project conducts sentiment analysis on sports-related content to reveal public emotions surrounding major sports events, specifically the **UEFA Champions League** from 2017 to 2023. Using a **fine-tuned BERT model**, we analyze social media texts to identify emotions like **joy, excitement, sadness, anger, pride, relief**, and **nervousness**. The **BERT** model's performance is benchmarked against traditional approaches like **Logistic Regression** and **Random Forest**, showing clear improvements.

---

## **1. Data and Methodology**

### **1.1 Datasets**
Three datasets were used for this project:
1. **Google Emotions Dataset**: A labeled dataset of Reddit comments, filtered to include seven sports-relevant emotions.
2. **ChatGPT-Generated Dataset**: A synthetic dataset of 50 structured texts to validate BERT's performance.
3. **Sports Sentiment Dataset**: Social media data (tweets, headlines) focused on 12 UEFA Champions League teams, spanning 2017–2023.

### **1.2 Preprocessing**
- Filtered **seven core emotions** from the Google Emotions Dataset.
- Extracted social media texts mentioning 12 prominent teams, including **Real Madrid, Manchester United, Bayern Munich**, and others.
- Cleaned and tokenized the data, ensuring high-quality input for model training.

### **1.3 Models Developed**
- **Traditional Models**: Logistic Regression and Random Forest, using **TF-IDF vectorization**.
- **Fine-Tuned BERT**: A transformer-based model trained using Hugging Face’s library for sequence classification tasks.

---

## **2. Results**

### **2.1 Social Media Sentiment Trends**

![Uploading Figure 2.png…]()
**Figure 2**: Frequency of social media texts related to UEFA teams from 2017 to 2023.

Social media engagement is dominated by **Premier League teams** such as **Manchester United** and **Liverpool**. **Championship-winning teams** experience spikes in activity, as seen with **Real Madrid in 2017, 2018, and 2022**, and **Bayern Munich in 2020**. However, teams from Italy's Serie A, like **Juventus** and **Inter Milan**, show comparatively lower engagement.

---

### **2.2 Model Comparison**

The performance of the three models reveals BERT's clear superiority:
- **Logistic Regression Accuracy**: 60%  
- **Random Forest Accuracy**: 58%  
- **BERT Accuracy**: 65%

![Uploading Figure 3.png…]()
**Figure 3**: Macro F1-Score comparison of models.

BERT achieves the highest F1-score and accuracy, outperforming Logistic Regression and Random Forest. Its deep contextual understanding enables better classification of nuanced emotions, particularly for challenging categories like **anger** and **nervousness**.

---

### **2.3 BERT Performance on ChatGPT-Generated Texts**

| Emotion      | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Anger        | 1.00      | 0.50   | 0.67     | 4       |
| Excitement   | 0.86      | 0.75   | 0.80     | 8       |
| Joy          | 0.67      | 0.86   | 0.75     | 7       |
| Nervousness  | 0.67      | 0.80   | 0.73     | 10      |
| Pride        | 0.80      | 0.67   | 0.73     | 6       |
| Relief       | 1.00      | 0.33   | 0.50     | 3       |
| Sadness      | 0.79      | 0.92   | 0.85     | 12      |
| **Accuracy** |           |        | **0.76** | 50      |
| **Macro Avg**| 0.83      | 0.69   | 0.72     | 50      |
| **Weighted Avg** | 0.79  | 0.76   | 0.75     | 50      |

**Table 4**: Classification performance of BERT on ChatGPT-generated texts.

When tested on well-structured synthetic data, BERT achieves an accuracy of **76%**. The results indicate strong performance, with high precision for emotions like **relief** and **sadness**. This shows that BERT generalizes well across both structured and unstructured text, making it versatile for real-world applications.

---

### **2.4 Emotion Proportions Over Time**

![Uploading Figure 4.png…]()

**Figure 4**: *Frequency of each predicted emotion in sports sentiment dataset (2017–2023).*

The proportions of emotions remain consistent across the years, reflecting the stability of the BERT model. **Joy** and **relief** are the most prominent emotions, capturing fan reactions during intense matches. Notably, while **sadness** and **anger** fluctuate slightly, they do not show significant long-term trends. This consistency suggests that fans' emotional responses follow predictable patterns during the UEFA Champions League seasons.

---

### **2.5 Sentiment Analysis for Championship Teams**


**Figure 5**: *Proportional changes in emotions for championship-winning teams (2017–2023).*

The graph shows how winning the championship impacts team sentiment compared to the yearly average:
- **Excitement** and **joy** typically increase but not consistently. For example, **Bayern Munich (2020)** saw lower-than-expected joy due to external factors like the Ballon d’Or cancellation.
- **Anger** and **sadness** often rise due to controversies or unmet expectations, as seen with **Chelsea (2021)**.
This highlights that championship success does not always guarantee a positive sentiment environment, as external narratives heavily influence fan emotions.

---

### **2.6 Emotion Distribution Across Teams**

![Figure 6](https://github.com/user-attachments/assets/632ac9ea-22a3-41ce-b470-35d2522926e3)

**Figure 6**: *Proportions of predicted emotions for different UEFA teams (2017–2023).*

The figure compares the emotional distributions for 12 UEFA teams:
- **Real Madrid** has a high proportion of **anger** and low **joy**, reflecting the demanding expectations of its fanbase despite consistent success.
- **Manchester United** and **Liverpool** show balanced distributions with high levels of **joy** and **excitement**, aligning with their widespread popularity.
- Teams like **Inter Milan** and **Juventus** exhibit unique emotional patterns, with higher proportions of **sadness** and **nervousness**.
These findings suggest that fan expectations and team culture significantly shape public sentiment beyond match outcomes.

---

## **3. Conclusion**
This study highlights the power of fine-tuned BERT models in sports sentiment analysis:
- **BERT outperformed Logistic Regression and Random Forest** with a 65% accuracy on sports-related content.
- Social media sentiment surrounding teams is complex, driven by factors beyond simple wins or losses.
- Teams like **Real Madrid** experience a unique emotional dynamic, reflecting high fan expectations and criticism despite consistent success.

The findings offer valuable insights for sports organizations to better understand **fan sentiment**, improve **engagement strategies**, and manage their public image.

---

## **4. Limitations and Future Work**
- **Limitations**: This study focused on seven predefined emotions and UEFA Champions League teams only.
- **Future Directions**:
  - Expand the dataset to include other sports and leagues.
  - Integrate multilingual analysis for cross-cultural insights.
  - Implement **real-time sentiment tracking** for live events.

---

## **5. Contact**
For inquiries, please contact:  
- **Hongyi Duan**: [h.duan@duke.edu](mailto:h.duan@duke.edu)  
- **Mu Niu**: [mu.niu@duke.edu](mailto:mu.niu@duke.edu)  
- **Zihan Xiao**: [zihan.xiao@duke.edu](mailto:zihan.xiao@duke.edu)

Here is the citation section formatted like the example for your **Sentiment Analysis Project**:

---

Here is the formatted **References** section for your README, following the style from the image:

---

## **References**

1. Demszky, G., Ghosh, D., Guha, A., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Retrieved from [https://aclanthology.org/2020.acl-main.372](https://aclanthology.org/2020.acl-main.372)

2. Gjurovic, T., et al. (2018). Reddit: A Gold Mine for Personality Prediction? Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. Retrieved from [https://aclanthology.org/W18-1112/](https://aclanthology.org/W18-1112/)

3. Hada, K., et al. (2021). Rudditt: Norms of Offensiveness for English Reddit Comments. Proceedings of the 2021 Annual Meeting of the Association for Computational Linguistics. Retrieved from [https://aclanthology.org/2021.acl-long.210/](https://aclanthology.org/2021.acl-long.210/)

4. Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint. Retrieved from [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)

5. Patel, S., & Passi, A. (2020). Sentiment Analysis on Twitter Data of World Cup Soccer Tournament Using Machine Learning. MDPI. Retrieved from [https://www.mdpi.com/2624-831X/1/2/14](https://www.mdpi.com/2624-831X/1/2/14)

6. Rothe, S., Narayan, S., & Severyn, A. (2019). Leveraging Pre-trained Checkpoints for Sequence Generation Tasks. arXiv preprint. Retrieved from [https://arxiv.org/abs/1907.12461](https://arxiv.org/abs/1907.12461)

---

## **Citation**

If you use this code or method in your work, please cite the report:

```bibtex
@misc{duan2024sportsbert,
  author  = {Hongyi Duan, Mu Niu, and Zihan Xiao},
  title   = {Sentiment Analysis of Sports-Related Content Using Fine-Tuned BERT Model},
  year    = {2024}
}
```


