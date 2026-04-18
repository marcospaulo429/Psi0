# Fine-tuning GR00T N1.6 no Psi0

Este guia descreve como **configurar e tunar** o treino (fine-tune) do baseline **GR00T N1.6** neste repositório. O fluxo oficial usa o launcher `finetune_gr00t.py` e presets YAML em `presets/train/`.

> O `README.md` na raiz do Psi0 ainda menciona `cd src/gr00t` e `./scripts/train_gr00t.sh`; no código atual o caminho canónico é **`baselines/gr00t-n1.6/`**.

---

## 1. Ambiente

O treino invoca explicitamente o Python do venv em **`src/gr00t/.venv`**:

```bash
cd /caminho/Psi0/src/gr00t
uv sync
cd /caminho/Psi0
```

Confirme que existe `src/gr00t/.venv/bin/python`.

Pode lançar o preset com `uv run python3 ...` a partir da raiz do Psi0 (usa o `pyproject.toml` da raiz) ou com `python3` se o ambiente tiver as dependências necessárias para o launcher.

---

## 2. Checkpoint base (obrigatório: pasta local)

`launch_finetune.py` lê **`base_model_path/config.json`** no disco. O identificador Hugging Face **`nvidia/GR00T-N1.6-3B` sozinho não funciona** como caminho.

1. Descarregue o modelo para um diretório local, por exemplo:

   ```bash
   hf download nvidia/GR00T-N1.6-3B --local-dir /caminho/Psi0/checkpoints/GR00T-N1.6-3B
   ```

2. Verifique que existe `.../GR00T-N1.6-3B/config.json`.

3. No preset YAML, use esse caminho absoluto em `model.base_model_path`, ou na CLI:

   ```bash
   --base-model-path /caminho/Psi0/checkpoints/GR00T-N1.6-3B
   ```

---

## 3. Lançar o treino

Na **raiz do repositório** Psi0:

```bash
python3 baselines/gr00t-n1.6/finetune_gr00t.py \
  --preset finetune_simple \
  --dry-run
```

Remova `--dry-run` para executar. Alternativa:

```bash
bash baselines/gr00t-n1.6/pretrain_gr00t.sh --preset pretrain_g1_ee --dry-run
```

### Argumentos úteis da CLI (`finetune_gr00t.py`)

| Flag | Efeito |
|------|--------|
| `--preset` | Nome do preset (sem `.yaml`) ou **caminho absoluto** para um ficheiro YAML. |
| `--dataset-path` | Sobrescreve `dataset.path` do preset. |
| `--base-model-path` | Sobrescreve `model.base_model_path`. |
| `--output-dir` | Sobrescreve `training.output_dir`. |
| `--cuda-visible-devices` | Ex.: `0` ou `0,1`. |
| `--num-gpus` | Alinha com o número de GPUs visíveis. |
| `--master-port` | Porta do `torchrun`. |
| `--dry-run` | Imprime o comando e sai. |

O batch global **não** tem flag dedicada; ajuste no YAML (secção `training`).

---

## 4. Estrutura dos presets YAML

Os presets estão em **`baselines/gr00t-n1.6/presets/train/`**.

- **`extends`**: herda de outro YAML (ex.: `_common_finetune.yaml` define hiperparâmetros partilhados).
- **`launcher`**: normalmente `launch_finetune` (definido em `_common_finetune.yaml`).
- **`model.base_model_path`**: diretório local do checkpoint (com `config.json`).
- **`dataset.path`**: raiz do dataset **LeRobot** (pasta com `meta/`, `data/`, `videos/`, …).
- **`dataset.embodiment_tag`**: enum `EmbodimentTag` (ex.: `G1_LOCO_DOWNSTREAM`).
- **`dataset.modality_config_path`**: módulo Python sob `src/gr00t/gr00t/configs/modality/` que regista a modalidade.
- **`training`**: `num_gpus`, `output_dir`, `global_batch_size`, `dataloader_num_workers`, `max_steps`, `learning_rate`, `save_steps`, `save_total_limit`, `warmup_ratio`, `weight_decay`, …
- **`runtime`**: `cuda_visible_devices`, `master_port`, opcionalmente `nproc_per_node`.
- **`augmentation.color_jitter`**: passado como `--color-jitter-params` para o `launch_finetune`.
- **`env`**: variáveis do subprocesso de treino.

Valores comuns partilhados: **`baselines/gr00t-n1.6/presets/train/_common_finetune.yaml`**.

---

## 5. `DATASET_PATH` e modalidades dinâmicas

Ficheiros como **`g1_locomanip.py`**, **`g1_ee.py`**, etc., leem **`DATASET_PATH`** do ambiente para carregar **`meta/modality.json`** do dataset.

Por isso:

1. Em presets que usam esses módulos, defina em **`env:`** o mesmo diretório que em **`dataset.path`**.
2. Se usar `--dataset-path` na CLI, o **`env.DATASET_PATH` no YAML** deve **continuar igual** ao dataset real (o launcher não sincroniza automaticamente os dois).

Exemplo mínimo:

```yaml
dataset:
  path: /abs/path/ao/dataset_lerobot
  embodiment_tag: G1_LOCO_DOWNSTREAM
  modality_config_path: src/gr00t/gr00t/configs/modality/g1_locomanip.py
env:
  DATASET_PATH: /abs/path/ao/dataset_lerobot
```

---

## 6. Escolher preset / modalidade por tipo de dados

### Dados reais Psi0 (LeRobot com `states` / `action` fatiados, vídeo egocêntrico como `rs_view`)

O `meta/modality.json` publicado para tarefas G1 reais costuma alinhar com **`g1_locomanip.py`** (chaves de estado/ação incluindo `torso_*`, `target_yaw`, vídeo `rs_view` → `observation.images.egocentric`).

Use:

- **`embodiment_tag`:** `G1_LOCO_DOWNSTREAM`
- **`modality_config_path`:** `src/gr00t/gr00t/configs/modality/g1_locomanip.py`

### Outros presets

- **`pretrain_g1_ee.yaml`**: fluxo HE / SIMPLE com **`g1_ee.py`** e `G1_EE_A16` — exige `modality.json` compatível com esse módulo (não é o layout típico do zip `psi-data` real acima).

Consulte `src/gr00t/gr00t/configs/modality/*.py` e compare com o vosso `meta/modality.json`.

---

## 7. Tunar hiperparâmetros

### Batch

- **`training.global_batch_size`**: batch **global** efetivo. Com `num_gpus: 1`, o batch por dispositivo é `global_batch_size // num_gpus`.
- **`global_batch_size` deve ser divisível por `num_gpus`** (ver `experiment.py`).

### Data loading

- **`training.dataloader_num_workers`**: `0` reduz memória e processos auxiliares; valores altos + cache de shards aumentam uso de RAM/VRAM em alguns setups.

### Passos, LR, regularização

Ajuste em `training` no YAML (ou sobrescreva campos herdados de `_common_finetune.yaml`):

- `max_steps`, `learning_rate`, `weight_decay`, `warmup_ratio`
- `save_steps`, `save_total_limit`

### Argumentos extra do `FinetuneConfig`

Chaves do dataclass em `src/gr00t/gr00t/configs/finetune_config.py` que **não** estão no mapa fixo de `finetune_gr00t.py` podem ser passadas via **`extra_args`** no preset:

```yaml
extra_args:
  gradient_accumulation_steps: 4
  num_shards_per_epoch: 2048
```

`num_shards_per_epoch` menor pode aliviar pressão quando a VRAM é limitada (comentário no próprio `FinetuneConfig`).

### Augmentação de cor

Definida em **`augmentation.color_jitter`** no preset (herdada ou sobrescrita).

### Flags de modelo (finetune “onde” treinar)

Passíveis de ir em **`extra_args`** se existirem no `FinetuneConfig`: por exemplo `tune_llm`, `tune_visual`, `tune_projector`, `tune_diffusion_model`, `reinit_action_head`, etc. O `launch_finetune.py` aplica defaults (ex.: `tune_llm=False`, `tune_visual=False` por omissão no fluxo finetune).

---

## 8. Memória GPU (VRAM) e DeepSpeed

- O log de treino pode mostrar **~1.6B+ parâmetros treináveis** na cabeça de acção; o otimizador **Adam** reserva tensores adicionais (momentum / variance).
- Em **`experiment.py`**, o ficheiro DeepSpeed só é usado quando **`num_gpus > 1`** e **`use_ddp`** é falso. Com **uma GPU**, não há ZeRO ativo por esse caminho — a VRAM tende a ser o maior gargalo.
- **Recomendações práticas:**
  - Use uma **GPU dedicada** (`CUDA_VISIBLE_DEVICES`) sem outros processos longos na mesma placa.
  - Para o mesmo modelo, **várias GPUs** (`num_gpus: 2`, `cuda_visible_devices: "0,1"`) podem ativar DeepSpeed e repartir estados do otimizador.
  - `global_batch_size` baixo ajuda no pico de activações, mas **não** resolve sozinho o pico do otimizador sobre ~1.6B parâmetros.
  - Opcional: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` para fragmentação (ajuda marginal).

O `launch_finetune.py` fixa `config.training.optim = "adamw_torch"`. Para otimizadores mais leves seria necessário alterar código ou estender o launcher.

---

## 9. Dados Hugging Face e metadados

Para datasets `USC-PSI-Lab/psi-data` em zip:

1. Instale a CLI: `pip install "huggingface_hub[cli]"` (comando `hf`).
2. Descarregue e descomprima conforme o README principal do Psi0.
3. Execute o patch de Parquet se aplicável:

   ```bash
   python scripts/data/patch_lerobot_meta.py /abs/path/ao/dataset_tarefa
   ```

---

## 10. Ficheiros de referência

| Ficheiro | Papel |
|----------|--------|
| `baselines/gr00t-n1.6/finetune_gr00t.py` | Monta `torchrun` + argumentos a partir do YAML. |
| `src/gr00t/gr00t/experiment/launch_finetune.py` | CLI `tyro` + carrega `config.json` do checkpoint local. |
| `src/gr00t/gr00t/configs/finetune_config.py` | Todos os campos tunáveis do job finetune. |
| `src/gr00t/gr00t/experiment/experiment.py` | Batch por dispositivo, DeepSpeed condicionado a `num_gpus`. |
| `src/gr00t/gr00t/configs/modality/*.py` | Registo de modalidade + validação contra `modality.json`. |

---

## 11. Avaliação e deploy

Ver `baselines/gr00t-n1.6/README.md`: `eval_simple.py`, `deploy_gr00t_simple.sh`, `sim_eval.sh`.
