# -*- coding: utf-8 -*-

import os
import sys
from PyPDF2 import PdfReader, PdfWriter
from .core import TableList
from .parsers import Stream, Lattice
from .utils import (
    TemporaryDirectory,
    get_page_layout,
    get_text_objects,
    get_rotation,
    is_url,
    download_url,
)

class PDFHandler(object):
    """Handles all operations like temp directory creation, splitting
    file into single page PDFs, parsing each PDF and then removing the
    temp directory.

    Parameters
    ----------
    filepath : str
        Filepath or URL of the PDF file.
    pages : str, optional (default: '1')
        Comma-separated page numbers.
        Example: '1,3,4' or '1,4-end' or 'all'.
    password : str, optional (default: None)
        Password for decryption.
    """

    def __init__(self, filepath, pages="1", password=None):
        if is_url(filepath):
            filepath = download_url(filepath)
            self.filepath = filepath
        else:
            self.filepath = filepath
        if not filepath.lower().endswith(".pdf"):
            raise NotImplementedError("File format not supported")

        if password is None:
            self.password = ""
        else:
            self.password = password
        if sys.version_info[0] < 3:
            self.password = self.password.encode("ascii")
        self.pages = self._get_pages(self.filepath, pages)

    def _get_pages(self, filepath, pages):
        """Converts pages string to list of ints.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        pages : str, optional (default: '1')
            Comma-separated page numbers.
            Example: '1,3,4' or '1,4-end' or 'all'.

        Returns
        -------
        P : list
            List of int page numbers.
        """
        page_numbers = []
        if pages == "1":
            page_numbers.append({"start": 1, "end": 1})
        else:
            instream = open(filepath, "rb")
            infile = PdfReader(instream)
            if infile.is_encrypted:
                infile.decrypt(self.password)
            if pages == "all":
                page_numbers.append({"start": 1, "end": len(infile.pages)})
            else:
                for r in pages.split(","):
                    if "-" in r:
                        a, b = r.split("-")
                        if b == "end":
                            b = len(infile.pages)
                        page_numbers.append({"start": int(a), "end": int(b)})
                    else:
                        page_numbers.append({"start": int(r), "end": int(r)})
            instream.close()
        P = []
        for p in page_numbers:
            P.extend(range(p["start"], p["end"] + 1))
        return sorted(set(P))

    def save_page(self, filepath, page, tempdir):
        """Saves specified page P from PDF into a temporary directory.

        Parameters
        ----------
        filepath : str
            Filepath or URL of the PDF file.
        page : int
            Page number.
        tempdir : str
            Tmp directory.
        """
        with open(filepath, "rb") as fileobj:
            infile = PdfReader(fileobj)
            if infile.is_encrypted:
                infile.decrypt(self.password)
            fpath = os.path.join(tempdir, "page-{0}.pdf".format(page))
            root, fext = os.path.splitext(fpath)
            p = infile.pages[page - 1]
            outfile = PdfWriter()
            outfile.add_page(p)
            with open(fpath, "wb") as f:
                outfile.write(f)
            layout, dim = get_page_layout(fpath)
            # fix rotated PDF
            chars = get_text_objects(layout, ltype="char")
            horizontal_text = get_text_objects(layout, ltype="horizontal_text")
            vertical_text = get_text_objects(layout, ltype="vertical_text")
            rotation = get_rotation(chars, horizontal_text, vertical_text)
            if rotation in [90, 180, 270]:
                fpath_new = "{0}.rotated{1}{2}".format(root, rotation, fext)
                os.rename(fpath, fpath_new)
                with open(fpath_new, "rb") as f:
                    infile = PdfReader(f)
                    if infile.is_encrypted:
                        infile.decrypt(self.password)
                    outfile = PdfWriter()
                    if rotation == 90:
                        p.rotate_clockwise(90)
                    elif rotation == 180:
                        p.rotate_clockwise(180)
                    elif rotation == 270:
                        p.rotate_counter_clockwise(90)
                    outfile.add_page(p)
                    with open(fpath, "wb") as f:
                        outfile.write(f)
                os.remove(fpath_new)
            instream.close()

    def parse(
        self, flavor="lattice", suppress_stdout=False, layout_kwargs={}, **kwargs
    ):
        """Extracts tables by calling parsers.get_tables on all single
        page PDFs.

        Parameters
        ----------
        flavor : str (default: 'lattice')
            The parsing method to use ('lattice' or 'stream').
            Lattice is used by default.
        suppress_stdout : str (default: False)
            Suppress logs and warnings.
        layout_kwargs : dict, optional (default: {})
            A dict of `pdfminer.layout.LAParams` kwargs.
        kwargs : dict
            See camelot.read_pdf kwargs.

        Returns
        -------
        tables : camelot.core.TableList
            List of tables found in PDF.
        """
        tables = []
        with TemporaryDirectory() as tempdir:
            for p in self.pages:
                self.save_page(self.filepath, p, tempdir)
            pages = [
                os.path.join(tempdir, "page-{0}.pdf".format(p)) for p in self.pages
            ]
            parser = Lattice(**kwargs) if flavor == "lattice" else Stream(**kwargs)
            for p in pages:
                t = parser.extract_tables(
                    p, suppress_stdout=suppress_stdout, layout_kwargs=layout_kwargs
                )
                tables.extend(t)
        return TableList(sorted(tables))
