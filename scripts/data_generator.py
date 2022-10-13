"""This module implements a generator of examples."""
import sys
sys.path.append('.')
from typing import List, Tuple, Iterator, Literal
from scripts.configs.config import Config
import sentencepiece as spm

from random import shuffle
from random import randrange


Sequence = List[str]
EncodedSequence = List[int]
Batch = List[EncodedSequence]
Example = Tuple[EncodedSequence, EncodedSequence]

class DataGenerator:
    """
    A class for iterating over batches of examples.
    
    Parameters
    ----------
    - config: Config
        An instance of the config class containing\
        all the relevant informations.
    """

    def __init__(self, config: Config, corpus_type: Literal["train", "dev"]):
        self.tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer)
        self.config = config
        corpus = config.to_dict()[corpus_type]
        self.examples = self.make_examples(corpus)
        self.size = len(self.examples)

    def pad(self, batch: Batch) -> Batch:
        """
        Add the pad tokens to all the sequences\
        in order to have the same sequence length\
        in the batch.

        Parameters
        ----------
        - batch: Batch
            The batch to pad.
        """
        max_length = max(len(sequence) for sequence in batch)
        new_batch = []
        for encoded_sequence in batch:
            new_batch.append(encoded_sequence + ([self.config.pad_idx] * (max_length - len(encoded_sequence))))
        return new_batch
            

    def encode(self,
               sequence: Sequence,
               add_bos: bool=True,
               add_eos: bool=True) -> EncodedSequence:
        """
        Encode a sequence of string tokens into integers.

        Parameters
        ----------
        - sequence: List of str
            Sequence to encode.
        - add_bos: bool
            Whether or not add the tokens for the beginning of the sequence.
        - add_eos: bool
            Whether or not add the tokens for the ending of the sequence.
        
        Returns
        -------
        - List of integer:
            The encoded sequence.
        """
        return self.tokenizer.encode(sequence,
                                     add_bos=add_bos,
                                     add_eos=add_eos)

    def decode(self, encoded_sequence: EncodedSequence) -> Sequence:
        """
        Decode a sequence of integer tokens into strings.

        Parameters
        ----------
        - encoded_sequence: List of int
            Sequence to decode.
        
        Returns
        -------
        - List of str:
            The decoded sequence.
        """
        return self.tokenizer.decode(encoded_sequence)


    def decode_subword(self, encoded_sequence: EncodedSequence) -> Sequence:
        """
        Decode a sequence of integer tokens into strings,\
        without merging subwords.

        Parameters
        ----------
        - encoded_sequence: List of int
            Sequence to decode.
        
        Returns
        -------
        - List of str:
            The decoded sequence.
        """
        return self.tokenizer.encode(encoded_sequence, out_type=str)

    def example(self, sequence: Sequence) -> Example:
        """
        Create a training example from a given sequence.
        A training example is a tuple of two sequences\
        of the same length. The first sequence of the tuple\
        is a encoded sequence with the BOS token but without\
        the EOS of token. The second sequence is the encoded\
        sequence with the EOS token but without the BOS token.

        Parameters
        ----------
        - sequence: List of str
            The sequence from which create a training example.
        
        Returns
        -------
        - Tuple:
            Tuple containing two sequence consisting of\
            the training example created from the sequence.
        """
        src, tgt = sequence.split(" @@@ ")
        src = src.strip()
        tgt = tgt.strip()
        return (self.encode(src),
                self.encode(tgt, add_eos=False),
                self.encode(tgt, add_bos=False))
    
    def make_examples(self, text_corpus: str) -> List[Example]:
        """
        Create examples from a given raw text corpus.

        Parameters
        ----------
        - text_corpus: str
            The text corpus from which extract examples.
        """

        examples = []
        with open(text_corpus) as text:
            for line in text:
                line = line.strip()
                if " @@@ " not in line:
                    continue
                example = self.example(line)
                if len(example[0]) > self.config.max_length:
                    print(len(example[0]))
                    continue
                examples.append(example)
        return examples

    def prompt(self) -> EncodedSequence:
        """Prompt an encoded sequence."""
        # choose randomly one example
        random_example = randrange(self.size)
        # get only the source sequence, not the target
        src, expected, _ = self[random_example]
        return src, expected

    def __getitem__(self, idx: int) -> Example:
        """
        Get an example a specific example.

        Parameters
        ----------
        - idx: int
            The specific example to get.
        """
        return self.examples[idx]

    def __call__(self, batch_size: int=32) -> Iterator[Batch]:
        """
        Batch generator.

        Parameters
        ----------
        - batch_size: int
            he size of the batches.
        
        Returns
        -------
        - Iterator:
            Iterator over the batches.
        """
        shuffle(self.examples)
        for step in range(0, self.size, batch_size) :
            src, tgt_x, tgt_y = zip(*self.examples[step:step + batch_size])
            yield self.pad(src), self.pad(tgt_x), self.pad(tgt_y)
