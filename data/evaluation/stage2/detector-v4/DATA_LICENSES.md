# Data and model licences used by detector v4

| Item | Source | Licence | Used for | Attribution / obligations |
|---|---|---|---|---|
| SDG-Warehouse v2 COCO frames | NVIDIA PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes (revision `d5b88d3a…`), frames built in `work/stage1/compact-detector/v2` | OpenMDW-1.1 | training (34 runs), selection (8 runs) | keep the OpenMDW-1.1 notice with redistributed data or weights derived from it; credit NVIDIA |
| SmartSpaces MTMC_Tracking_2026 | `nvidia/PhysicalAI-SmartSpaces` on Hugging Face (frames from `detector-v3/data/ss_frames`) | CC BY 4.0 | training (Warehouse_000/002/004/006/008/009), selection (012–016), re-used test (010/018/019/022) | attribution to NVIDIA, licence link, indicate changes (frames resized to 640×360, tiny boxes grey-filled) |
| LOCO v1 (Logistics Objects in Context) | github.com/tum-fml/loco; 5,097 annotated images fetched by v3 (`detector-v3/data/loco`) | CC0-1.0 (public-domain dedication) | training (subsets 1, 2, 4, 5 except validation sessions), selection (3 sessions), test (subset 3) | none required; the authors ask for citation: Mayershofer, Holm, Molter, Fottner, "LOCO: Logistics Objects in Context", ICMLA 2020 |
| keremberke/forklift-object-detection | Hugging Face dataset, revision `2c0f4e8e9085f3079b80137e780aab0e54936cfc`; mirror of Roboflow Universe "forklift-dsitv" v1 by Mohamed Traore (2022), images originally from images.cv | CC BY 4.0 (stated in the Roboflow export and in every image's COCO licence field) | training (train 295), selection (valid 84), test (test 42) | attribution required in any redistribution or derived-model documentation: "Forklift Dataset, Mohamed Traore, Roboflow Universe, https://universe.roboflow.com/mohamed-traore-2ekkp/forklift-dsitv, CC BY 4.0"; indicate changes (ignore regions grey-filled in training copies) |
| YOLOX code | Megvii-BaseDetection/YOLOX, commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee` (`work/stage1/compact-detector/YOLOX`) | Apache-2.0 | model definition, SimOTA loss, transforms | keep the Apache-2.0 licence and NOTICE with redistributed code |
| YOLOX-tiny COCO checkpoint | YOLOX release 0.1.1rc0 `yolox_tiny.pth`, sha256 `9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0` | Apache-2.0 | initialisation of arm T | as above |
| YOLOX-s COCO checkpoint | YOLOX release 0.1.1rc0 `yolox_s.pth`, sha256 `f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30` | Apache-2.0 | initialisation of arm S | as above |
| YOLOX-L COCO checkpoint | YOLOX release 0.1.1rc0 `yolox_l.pth`, sha256 `1e6b7fa6240375370b2a8a8eab9066b3cdd43fd1d0bfa8d2027fd3a51def2917` | Apache-2.0 | person pseudo-labels (teacher) on LOCO training/validation sites and keremberke train/valid | as above |

Notes

- COCO-pretrained checkpoints were trained by their authors on MS COCO; COCO annotations are CC BY 4.0 and its images carry their Flickr licences. We use only the released weights, as v1–v3 did.
- A v4 model trained on these data is a derived work of CC BY 4.0 datasets (SmartSpaces, keremberke); a public release of the weights should carry the attribution lines above. LOCO adds no obligation.
- No data was uploaded anywhere; downloads in this study: keremberke zips (20 MB) and the YOLOX-s / YOLOX-L checkpoints (72 MB, 434 MB) into `data/downloads/`.
