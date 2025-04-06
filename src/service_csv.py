import csv
from collections import Counter
from os.path import dirname

from src.utils import create_dir_if_not_exits


class CsvService:

    @staticmethod
    def write(target, lines_it, header=None, notify=True, force_mk_dir=False, delimiter="\t", quotechar='"'):
        assert(isinstance(header, list) or header is None)

        if force_mk_dir:
            create_dir_if_not_exits(dirname(target))

        counter = Counter()
        with open(target, "w") as f:
            w = csv.writer(f, delimiter=delimiter, quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)

            if header is not None:
                w.writerow(header)

            for content in lines_it:
                w.writerow(content)
                counter["written"] += 1

        if notify:
            print(f"Saved: {target}")
            print("Total rows: {}".format(counter["written"]))

    @staticmethod
    def read(target, delimiter='\t', quotechar='"', skip_header=False, cols=None, return_row_ids=False,
             open_func=None):
        assert(isinstance(cols, list) or cols is None)

        header = None

        with open(target, newline='\n') if open_func is None else open_func(target) as f:
            for row_id, row in enumerate(csv.reader(f, delimiter=delimiter, quotechar=quotechar)):
                if skip_header and row_id == 0:
                    header = row
                    continue

                # Determine the content we wish to return.
                if cols is None:
                    content = row
                else:
                    row_d = {header[col_name]: value for col_name, value in enumerate(row)}
                    content = [row_d[col_name] for col_name in cols]

                # Optionally attach row_id to the content.
                yield [row_id] + content if return_row_ids else content
