# fake-news-detection-group2

Problem Statement
False or misleading claims spread rapidly across social media and
news platforms, often appearing as short, context-poor snippets.
This project aims to estimate the truthfulness of short political
statements using the LIAR dataset, which categorizes claims into six
labels ranging from pants-on-fire to true. The task presents several
challenges: (1) the inputs are brief, offering limited textual evidence;
(2) the labels are fine-grained, making it easy to confuse neighboring
classes such as half-true and mostly-true; and (3) the dataset includes
metadata (e.g., speaker, party, state, subject) that may introduce
biases, allowing models to rely on contextual cues rather than the
content itself.
To ensure fair evaluation, we begin with transparent text-only base-
lines using TF-IDF features and simple linear models. We then ana-
lyze model behavior through two diagnostic tests: a last-sentence
ablation to assess dependence on specific phrases, and a metadata-
only baseline to measure performance without textual input. These
steps help isolate genuine linguistic signals from dataset artifacts.
Finally, we extend the setup to a transformer-based model (BERT)
under identical data splits and evaluation metrics, enabling a direct
comparison between traditional feature-based methods and modern
neural encoders. Although BERT improves overall performance,
the fine-grained nature of the labels remains a key challenge, and
robustness checks remain essential to ensure reliable interpretation.
