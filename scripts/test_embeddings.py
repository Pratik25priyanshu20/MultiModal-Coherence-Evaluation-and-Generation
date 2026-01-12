from src.coherence.msci import compute_msci_v0
from src.embeddings.aligned_embeddings import AlignedEmbedder


def main() -> None:
    embedder = AlignedEmbedder(target_dim=512)

    text = "A calm forest road at dawn with soft fog and birds chirping"
    image_path = "data/processed/images/sample.jpg"
    audio_path = "data/processed/audio/sample.wav"

    emb_text = embedder.embed_text(text)
    emb_image = embedder.embed_image(image_path)
    emb_audio = embedder.embed_audio(audio_path)

    msci = compute_msci_v0(emb_text, emb_image, emb_audio)
    print(msci)


if __name__ == "__main__":
    main()
