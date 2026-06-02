Retrieval

python rag4vln/scripts/eval/eval_retriever.py --dataset-json data/vln_ce/raw_data_mask_1/r2r/val_seen/val_seen.json --gt-csv data/vln_ce/dataset_gt.csv --text-embedder bge --vision-embedder vit --topk1 3 --topk2 3 --topk3 10 --hit-k 5 --max-episodes 150 --no-export-images --kb-embed-cache rag4vln/results/cache/kb_embed_bge_vit.pt --subset-name val_seen_mask_1

python rag4vln/scripts/eval/eval_retriever_baseline.py --baseline global --dataset-json data/vln_ce/raw_data_mask_1/r2r/val_seen/val_seen.json --gt-csv data/vln_ce/dataset_gt.csv --subset-name mask_bge_global --text-embedder bge --vision-embedder vit --topk1 3 --topk2 3 --topk3 10 --hit-k 5 --max-episodes 0 --kb-embed-cache rag4vln/results/cache/kb_embed_bge_vit_baseline.pt

python rag4vln/scripts/eval/grid_search_alpha_beta.py --start 0 --end 10 --step 1 --hit-k 5

python rag4vln/scripts/eval/eval_rag4vln_vln_augmented.py --config rag4vln/scripts/eval/configs/habitat_dual_system_cfg.py --text-embedder bge --vision-embedder vit --kb-embed-cache rag4vln/results/cache/kb_embed_bge_vit.pt --max-episodes 300 --save-instruction-pairs --save-video

python rag4vln/scripts/eval/bench_rag4vln_latency.py --n-instructions 100 --output-txt rag4vln/results/my_latency.txt

python rag4vln/scripts/eval/eval_rag4vln_streamvln.py --model-path mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3 --episode-json-gz rag4vln/results/augmented_vln_eval/implicit/r_only/val_unseen_original.json.gz --eval-split val_unseen --nproc-per-node 1 --output-path rag4vln/results/StreamVLN/implicit/strvln_baseline