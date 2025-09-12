## Giới thiệu

Dự án Nhận diện thực thể y khoa (NER) để huấn luyện mô hình PhoBERT / XLM-R kết hợp BiLSTM + CRF để gán nhãn y khoa (Bệnh_lý, Triệu_chứng, Nguyên_nhân, Điều_trị, Phòng_ngừa, Tên_thuốc, Bộ_phận_cơ_thể, Chẩn_đoán).

## Cài đặt môi trường
```bash
conda create -n ner_env python=3.10 -y
conda activate ner_env

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install "transformers==4.41.0" "tokenizers==0.15.2" sentencepiece protobuf
pip install seqeval scikit-learn pandas matplotlib tqdm torchcrf
```
## Chuẩn bị dữ liệu
Bước 1: Chuẩn hóa & tách câu
```bash
$ python src/data/normalize_split_rebuild.py ^
  --input_jsonl data/raw/5sentencelabel.jsonl ^
  --out_dir data/processed/jsonl/split_5sent_clean
```

Bước 2: Thống kê phân bố nhãn
```bash
$ python src/data/label_distribution_report.py ^
  --data_dir data/processed/jsonl/split_5sent_clean ^
  --outdir outputs/figures
```

Bước 3: Huấn luyện NER


## XLM-R + BiLSTM-CRF
```bash
$ python src/train_span_bilstm_crf.py ^
  --data_dir data/processed/jsonl/split_5sent_clean ^
  --model_name_or_path xlm-roberta-base ^
  --output_dir models/xlmr-span-bilstm-crf ^
  --num_epochs 12 --batch_size 8 ^
  --lr 2e-5 --lr_lstm 1e-3 --weight_decay 0.01 --warmup_ratio 0.1 ^
  --max_len 512 ^
  --oversample_rare True --bucket_by_length True ^
  --snap_punct True --save_best True --seed 42 ^
  --amp True --num_workers 2 --pin_memory True
```
## PhoBERT + BiLSTM-CRF
```bash
$ python src/train_span_bilstm_crf.py ^
  --data_dir data/processed/jsonl/split_5sent_clean ^
  --model_name_or_path vinai/phobert-base ^
  --output_dir models/phobert-span-bilstm-crf ^
  --num_epochs 12 --batch_size 8 ^
  --lr 2e-5 --lr_lstm 1e-3 --weight_decay 0.01 --warmup_ratio 0.1 ^
  --max_len 256 ^
  --oversample_rare True --bucket_by_length True ^
  --snap_punct True --save_best True --seed 42 ^
  --amp True --num_workers 2 --pin_memory True
```

Bước 4: Inference
Inference trên test set
```bash
$ python src/infer_span.py ^
  --ckpt_dir models/xlmr-span-bilstm-crf/run_YYYYmmdd-HHMMSS ^
  --input_jsonl data/processed/jsonl/split_5sent_clean/test.jsonl ^
  --out_json outputs/pred_test.json
```
Hiển thị 

```bash
$ python src/viz/demo.py ^
  --pred_json outputs/pred_test.json ^
  --out_html outputs/pred_view.html
```