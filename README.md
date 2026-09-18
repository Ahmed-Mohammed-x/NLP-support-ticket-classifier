# NLP Support Ticket Classifier

This project evaluates three classical NLP models for predicting customer-support
ticket types from their subjects and descriptions.

## Run the notebook

1. Download or open `support_ticket_classifier.ipynb` in Google Colab.
2. Select **Runtime → Run all**.
3. Upload `customer_support_tickets.csv` when requested.
4. Wait for the model comparison and final evaluation to finish.

Required packages:

```bash
pip install pandas matplotlib scikit-learn
```

## Dataset

The project uses the Customer Support Ticket Dataset from Kaggle:

https://www.kaggle.com/datasets/waseemalastal/customer-support-ticket-dataset

The original dataset contains:

- 8,469 rows
- 17 columns
- Five ticket types
- No missing values in the required columns
- 279 duplicate description-label records

After duplicate removal, 8,190 rows remained.

## Methods

`Ticket Subject` and `Ticket Description` are combined into one text field and
represented using TF-IDF with unigrams and bigrams.

Three classifiers are compared using five-fold cross-validation macro F1:

1. Multinomial Naive Bayes
2. Logistic Regression
3. LinearSVC

The model with the highest cross-validation macro F1 is evaluated on a separate
20% test set.

## Results

| Model | CV macro F1 |
|---|---:|
| Multinomial Naive Bayes | 0.182 ± 0.009 |
| Logistic Regression | 0.188 ± 0.006 |
| LinearSVC | 0.191 ± 0.006 |

- Selected candidate: **LinearSVC**
- Test accuracy: **0.19**
- Test macro F1: **0.19**
- Strongest category: **Refund request — F1 0.22**
- Weakest category: **Cancellation request — F1 0.16**

## Conclusion

LinearSVC achieved the highest cross-validation score among the three candidate
models. However, its final accuracy and macro F1 were only 0.19.

The weak performance and widespread confusion between categories indicate that
the ticket subjects and descriptions are not sufficiently aligned with the
provided `Ticket Type` labels. The current classifier is therefore unsuitable
for dependable automatic ticket routing.

Possible improvements include auditing and relabelling the data, collecting
better-labelled historical tickets, and testing the model again after improving
label quality. Hybrid semantic retrieval could also be considered as an
alternative because it would find similar historical tickets without relying on
unreliable category labels.
