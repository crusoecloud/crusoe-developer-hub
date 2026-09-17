import argparse

import crusoe_tinker
import tinker
from transformers import AutoTokenizer

BASE_URL = "https://api.intelligence.crusoecloud.com"
PROMPT = "Hello, how are you?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--checkpoint-name", required=True)
    ap.add_argument("--base-model", required=True)
    args = ap.parse_args()

    placement = [crusoe_tinker.BaseModelConfig(base_model=args.base_model)]
    service_config = crusoe_tinker.ServiceConfig(
        training_client_base_models=placement,
        sampling_client_base_models=placement,
        context_len=8192,
    )
    sc = crusoe_tinker.ServiceClient(
        base_url=BASE_URL,
        api_key=args.api_key,
        service_config=service_config,
        max_retries=360,
    )
    rc = sc.create_rest_client()

    checkpoints = rc.list_checkpoints(training_run_id=args.session_id).result().checkpoints
    if not checkpoints:
        raise SystemExit(f"No checkpoints found on session {args.session_id!r}.")
    checkpoint = next((c for c in checkpoints if args.checkpoint_name in c.tinker_path), None)
    if checkpoint is None:
        names = ", ".join(c.tinker_path.rsplit("/", 1)[-1] for c in checkpoints)
        raise SystemExit(
            f"No checkpoint on session {args.session_id!r} matches name "
            f"{args.checkpoint_name!r}. Available: {names}"
        )

    sampler = sc.create_sampling_client(model_path=checkpoint.tinker_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    result = sampler.sample(
        prompt=tinker.ModelInput.from_ints(prompt_ids),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=100, temperature=0.7),
    ).result()
    print(tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
