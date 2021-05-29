# ADR_crosslingual
Курсовая работа, ФКН НИУ ВШЭ, Илья Пахалко БПМИ181

Курсовая работа по теме "Обработка биомедицинских текстов". Реализуется внедрение механизма учёта неуверенности модели-учителя для задачи распознавания сущностей в тексте (Named Entity Recognition, NER) для zero-shot моделей.

Zero-Shot learning: https://arxiv.org/abs/2004.12440

Few-Shot learning: https://arxiv.org/abs/2006.15315

ADAN: https://www.mitpressjournals.org/doi/pdfplus/10.1162/tacl_a_00039 + https://arxiv.org/pdf/1810.03552.pdf

Guide по запуску ноутбука (ADR_alpha.ipynb):

Прежде, чем запустить все ячейки, просмотрите (и, возможно, измените) ячейку с конфигом эксперимента.
Это первая ячейка в секции "Code". Ниже - небольшой обзор того, как им пользоваться:

```python
teacher_config = TrainConfig(
    model_type={
        'tokenizer': BertTokenizer,                   # Классы используемой архитектуры: токенайзера, конфига и модели из single_model.py
        'config': BertConfig,
        'model': BertTokenClassifier,
        'subword_prefix': '##',                       # сабтокенный префикс (если используется Bert) или суффикс (если используется XLM)
        #'subword_suffix': '<\w>',
    },
    model_checkpoint='bert-base-multilingual-cased',  # None или строка с загружаемым чекпоинтом модели
    optimizer_class=AdamW,                            # класс предпочитаемого оптимайзера...
    optimizer_kwargs={'lr':2e-5, 'eps':1e-8},         # и его параметры.

    train_batch_sz=4,                                 # размеры батчей..
    test_batch_sz=4,
    epochs=5                                          # число эпох для обучения
)

student_config = TrainConfig(                         # аналогично для модели-студента
    model_type={
        'tokenizer': BertTokenizer,
        'config': BertConfig,
        'model': BertTokenClassifier,
        'subword_prefix': '##',
        #'subword_suffix': '<\w>',
    },
    model_checkpoint='bert-base-multilingual-cased',
    optimizer_class=AdamW,
    optimizer_kwargs={'lr':2e-5, 'eps':1e-8},
    train_batch_sz=4,
    test_batch_sz=2,
    epochs=12
)

sampler_config = SamplerConfig(
    sampler_class=MarginOfConfidenceSampler,       # класс используемого семплера из samplers.py

    sampler_kwargs={'strategy':'confident',        # стратегия семплирования: 'confident' = отбирать семплы, в которых учитель уверен,
                                                   #'uncertain' - наоборот

                    'n_samples_out':student_config.train_batch_sz,},  # [не менять] сколько семплов отобрать из текущей подвыборки
                    #'n_forward_passes':10},          # параметр для стохастических семплеров (BALD и Variance):
                                                      # сколько проходов с дропаутом совершить, чтобы сгенерировать распределение для каждого семпла

    n_samples_in=10,                        #  входной размер подвыборки, из которой семплируем
)

exp_config = ExperimentConfig(
    teacher_config=teacher_config,
    student_config=student_config,
    sampler_config=sampler_config,
    n_few_shot=40,                         # сколько размеченных семплов будет использовано для донастройки учителя
    experiment_name="exp 40 shot, bert -> bert",  # произвольное название эксперимента
    seed=42,
    teacher_set='cadec'                    # 'cadec', 'psytar', или 'joined' - на каком датасете учим учителя
)
```
 
