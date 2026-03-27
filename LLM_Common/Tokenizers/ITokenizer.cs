using System;

namespace LLM.Tokenizers
{
    /// <summary>
    /// Strategy interface for tokenizer implementations.
    ///
    /// A tokenizer converts raw text into integer token IDs for the transformer
    /// and converts IDs back to human-readable text.
    ///
    /// Provided implementations:
    ///   <see cref="CharTokenizer"/>          – character-level (one ID per character)
    ///   <see cref="BpeTokenizer"/>           – Byte Pair Encoding        (GPT-2 / GPT-4 style)
    ///   <see cref="WordPieceTokenizer"/>     – WordPiece                 (BERT / DistilBERT style)
    ///   <see cref="SentencePieceTokenizer"/> – SentencePiece BPE + ▁    (LLaMA / Mistral style)
    ///   <see cref="UnigramTokenizer"/>       – Unigram Language Model    (T5 / ALBERT style)
    ///
    /// Usage – swap the tokenizer without changing any other code:
    ///
    ///   ITokenizer tok = new CharTokenizer(corpus);
    ///   ITokenizer tok = new BpeTokenizer(corpus, numMerges: 500);
    ///   ITokenizer tok = new WordPieceTokenizer(corpus, targetVocabSize: 1000);
    ///   ITokenizer tok = new SentencePieceTokenizer(corpus, numMerges: 500);
    ///   ITokenizer tok = new UnigramTokenizer(corpus, targetVocabSize: 1000);
    /// </summary>
    public interface ITokenizer
    {
        /// <summary>Total number of tokens in the vocabulary.</summary>
        int VocabSize { get; }

        /// <summary>Token ID returned for out-of-vocabulary input.</summary>
        int UnknownId { get; }

        /// <summary>Encode a text string into a sequence of integer token IDs.</summary>
        int[] Encode(string text);

        /// <summary>Decode a sequence of token IDs back to a text string.</summary>
        string Decode(int[] ids);

        /// <summary>
        /// Decode a single token ID to its surface-form string.
        /// Used by generation loops to stream tokens one at a time as they are sampled.
        /// </summary>
        string DecodeToken(int id);

        /// <summary>Print a vocabulary summary to stdout.</summary>
        void PrintVocab();

        /// <summary>
        /// Save the vocabulary to a file so it can be reloaded without the original corpus.
        /// The file should be named <c>{weightFile}.vocab</c> to keep the pair together.
        /// </summary>
        void SaveVocab(string path);
    }
}
