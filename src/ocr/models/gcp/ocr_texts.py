import json
from google.cloud.vision_v1.types import TextAnnotation


class OCRText:
    """
    Base class for all the text types in the OCR output
    """

    def __init__(self, text, confidence, bounding_box):
        self.text = text
        self.confidence = confidence
        self.bounding_box = bounding_box


class Symbol(OCRText):
    """
    Represents a single symbol in the OCR output
    """

    def __init__(self, symbol):
        super().__init__(symbol.text, symbol.confidence, symbol.bounding_box)


class Word(OCRText):
    def __init__(self, word):
        self.symbols = [Symbol(symbol) for symbol in word.symbols]
        assembled_word = ""
        for symbol in self.symbols:
            assembled_word += symbol.text
        super().__init__(assembled_word.strip(), word.confidence, word.bounding_box)

    def remove_symbol(self, symbol):
        if symbol in self.symbols:
            self.symbols.remove(symbol)


class Paragraph(OCRText):
    def __init__(self, paragraph):
        self.words = [Word(word) for word in paragraph.words]
        assembled_paragraph = ""
        for word in self.words:
            assembled_paragraph += word.text + " "
        super().__init__(
            assembled_paragraph.strip(),
            paragraph.confidence,
            paragraph.bounding_box,
        )

    def remove_word(self, word):
        if word in self.words:
            self.words.remove(word)

    def remove_symbol(self, symbol):
        for word in self.words:
            word.remove_symbol(symbol)
            if len(word.symbols) == 0:
                self.remove_word(word)


class Block(OCRText):
    def __init__(self, block):
        self.paragraphs = [Paragraph(paragraph) for paragraph in block.paragraphs]
        assembled_block = ""
        for paragraph in self.paragraphs:
            assembled_block += paragraph.text + "\n"
        super().__init__(
            assembled_block.strip(),
            block.confidence,
            block.bounding_box,
        )

    def remove_paragraph(self, paragraph):
        if paragraph in self.paragraphs:
            self.paragraphs.remove(paragraph)

    def remove_word(self, word):
        for paragraph in self.paragraphs:
            paragraph.remove_word(word)
            if len(paragraph.words) == 0:
                self.remove_paragraph(paragraph)

    def remove_symbol(self, symbol):
        for paragraph in self.paragraphs:
            paragraph.remove_symbol(symbol)
            if len(paragraph.words) == 0:
                self.remove_paragraph(paragraph)


def compare_texts(
    text_to_find, ocr_text, threshold, lower_case=False, substrings=False
):
    base_condition = (
        text_to_find == ocr_text.text.strip() and ocr_text.confidence >= threshold
    )
    if lower_case and not substrings:
        base_condition = (
            text_to_find.lower() == ocr_text.text.strip().lower()
            and ocr_text.confidence >= threshold
        )
    if substrings and not lower_case:
        base_condition = (
            # we also remove all spaces since OCR texts can sometimes be split by spaces
            text_to_find.replace(" ", "") in ocr_text.text.strip().replace(" ", "") and ocr_text.confidence >= threshold
        )
    if substrings and lower_case:
        base_condition = (
            # we also remove all spaces since OCR texts can sometimes be split by spaces
            text_to_find.lower().replace(" ", "") in ocr_text.text.strip().lower().replace(" ", "")
            and ocr_text.confidence >= threshold
        )
    return base_condition


class OCRTexts:
    """
    A class representing the hierarchy of texts in an OCR output
    """

    def __init__(self, document_fn=None, blocks=[]):
        if blocks:
            self.blocks = blocks
        else:
            self.document = TextAnnotation.from_json(json.load(open(document_fn)))
            self.blocks = [
                Block(block) for page in self.document.pages for block in page.blocks
            ]

    
    def remove_ocr_text(self, ocr_text):
        if isinstance(ocr_text, Block):
            self.remove_block(ocr_text)
        elif isinstance(ocr_text, Paragraph):
            self.remove_paragraph(ocr_text)
        elif isinstance(ocr_text, Word):
            self.remove_word(ocr_text)
        elif isinstance(ocr_text, Symbol):
            self.remove_symbol(ocr_text)
    
    
    def remove_block(self, block):
        if block in self.blocks:
            self.blocks.remove(block)

    def remove_paragraph(self, paragraph):
        for block in self.blocks:
            block.remove_paragraph(paragraph)
            if len(block.paragraphs) == 0:
                self.remove_block(block)

    def remove_word(self, word):
        for block in self.blocks:
            block.remove_word(word)
            if len(block.paragraphs) == 0:
                self.remove_block(block)

    def remove_symbol(self, symbol):
        for block in self.blocks:
            block.remove_symbol(symbol)
            if len(block.paragraphs) == 0:
                self.remove_block(block)

    def filter_blocks_with_little_text(
        self, num_of_words_in_block_threshold=2, len_of_label_key_threshold=3
    ):
        """
        Remove blocks that we suspect contains little information, as a heuristic that they are labels.
        We call this method when we want the OCR texts to contain mainly legend content.
        We identify non legend candidates by blocks containing little information and filter them out.
        """
        blocks_to_remove = []
        for block in self.blocks:
            block_text = block.text.replace("\n", " ")
            block_words = block_text.split()
            if len(block_words < num_of_words_in_block_threshold):
                # If the block contains a small amount of words - it's probably not part of a legend
                blocks_to_remove.append(block)
                continue
            label_candidates = [
                text
                for text in block_text.split()
                if len(text) < len_of_label_key_threshold
            ]
            if len(label_candidates) > 0.7 * len(block_words):
                # the block contains a lot of label-like candidates - it's probably not part of a legend
                blocks_to_remove.append(block)
        for block in blocks_to_remove:
            self.blocks.remove(block)

    def filter_blocks_with_text_chuncks(self):
        """
        Remove blocks that we suspect could be noise/legends.
        We check if the block contains words that aren't 'labels' (i.e. words of length > 3 as a heuristic).
        This can indicate that the block contains at least one legend row in the form of key: value.
        """
        blocks_to_remove = []
        for block in self.blocks:
            block_text = block.text.replace("\n", " ")
            non_label_candidates = [
                text for text in block_text.split() if len(text) > 3
            ]
            if non_label_candidates:
                # the block contains a lot of normal text that is probably not a label marking
                blocks_to_remove.append(block)
        for block in blocks_to_remove:
            self.blocks.remove(block)

    def get_paragraphs_with_text(
        self, text, threshold, lower_case=False, substrings=False
    ):
        return [
            paragraph
            for block in self.blocks
            for paragraph in block.paragraphs
            if compare_texts(text, paragraph, threshold, lower_case, substrings)
        ]

    def get_words_with_text(self, text, threshold, lower_case=False, substrings=False):
        return [
            word
            for block in self.blocks
            for paragraph in block.paragraphs
            for word in paragraph.words
            if text == word.text.strip() and word.confidence >= threshold
        ]

    def get_symbols_with_text(
        self, text, threshold, lower_case=False, substrings=False
    ):
        return [
            symbol
            for block in self.blocks
            for paragraph in block.paragraphs
            for word in paragraph.words
            for symbol in word.symbols
            if text == symbol.text.strip() and symbol.confidence >= threshold
        ]

    def find_texts(
        self,
        texts,
        threshold=0,
        check_symbols=False,
        filter_text_chuncks=True,
        lower_case=False,
        substrings=False,
    ):
        """
        Find a list of texts in the OCR output. The texts are removed from the OCR output.

        Args:
        texts (list): A list of texts to search for
        threshold (float): A threshold for the confidence of the text
        check_symbols (bool): Whether or not to to search for chars at the symbol level
        filter_text_chuncks (bool): Whether or not to filter out blocks that contain text chuncks,
        as a heuristic for filtering out legends

        Returns:
        list: A list of the texts found
        """
        if filter_text_chuncks:
            self.filter_blocks_with_text_chuncks()
        if not self.blocks:
            return []
        # sort the texts by length so that we first look for the longest texts.
        # this is because the longest texts are most likely to be unique and not a part of a longer text
        sorted_texts = sorted(texts, key=len, reverse=True)
        texts_found = []
        for text in sorted_texts:
            found = []
            paragraphs = self.get_paragraphs_with_text(text, threshold, lower_case, substrings)
            for paragraph in paragraphs:
                found.append(paragraph)
                self.remove_paragraph(paragraph)
            words = self.get_words_with_text(text, threshold, lower_case, substrings)
            for word in words:
                found.append(word)
                self.remove_word(word)
            if not found and check_symbols:
                # Search in symbols only if we haven't found any words/paragraphs
                symbols = self.get_symbols_with_text(text, threshold, lower_case, substrings)
                for symbol in symbols:
                    found.append(symbol)
                    self.remove_symbol(symbol)
            texts_found += found
        return texts_found

    def get_ocr_texts(self, level="paragraph", order="top_to_bottom"):
        """
        Get all the texts in the OCR output in OCRText format.
        Args:
            level (str): The level of the texts to get. Can be 'block', 'paragraph', 'word' or 'symbol'
            order (str): The order of the texts. Can be 'top_to_bottom' or 'left_to_right'
        Returns:
            list: A list of the texts
        """
        if level not in ["block", "paragraph", "word", "symbol"]:
            raise ValueError(
                f'Invalid level: {level}. Must be one of "block", "paragraph", "word" or "symbol"'
            )
        if order not in ["top_to_bottom", "left_to_right"]:
            raise ValueError(
                f'Invalid order: {order}. Must be one of "top_to_bottom" or "left_to_right"'
            )
        ocr_texts = []
        if level == "block":
            ocr_texts = [block for block in self.blocks]
        if level == "paragraph":
            ocr_texts = [
                paragraph for block in self.blocks for paragraph in block.paragraphs
            ]
        if level == "word":
            ocr_texts = [
                word
                for block in self.blocks
                for paragraph in block.paragraphs
                for word in paragraph.words
            ]
        if level == "symbol":
            ocr_texts = [
                symbol
                for block in self.blocks
                for paragraph in block.paragraphs
                for word in paragraph.words
                for symbol in word.symbols
            ]
        if order == "top_to_bottom":
            ocr_texts.sort(key=lambda paragraph: paragraph.bounding_box.vertices[0].y)
        elif order == "left_to_right":
            ocr_texts.sort(key=lambda paragraph: paragraph.bounding_box.vertices[0].x)
        return ocr_texts

    def get_texts(
        self,
        level="paragraph",
        order="top_to_bottom",
        filter_text_chuncks=False,
        filter_little_text=False,
    ):
        """
        Get all the texts in the OCR output.
        Args:
            level (str): The level of the texts to get. Can be 'block', 'paragraph', 'word' or 'symbol'
            order (str): The order of the texts. Can be 'top_to_bottom' or 'left_to_right'
        Returns:
            list: A list of the texts
        """
        if filter_text_chuncks:
            self.filter_blocks_with_text_chuncks()
        if filter_little_text:
            self.filter_blocks_with_little_text()
        ocr_texts = self.get_ocr_texts(level, order)
        return [text.text for text in ocr_texts]
