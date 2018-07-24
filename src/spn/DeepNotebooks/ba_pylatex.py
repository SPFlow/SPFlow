from pylatex.figure import Figure, SubFigure
from pylatex.utils import NoEscape, escape_latex

class TableFigure(Figure):
    latex_name = "figure"

    def add_table(self, table, *, width=NoEscape(r'0.8\textwidth'),
            placement=NoEscape(r'\centering')):
        """Add a table to the figure.
        Args
        ----
        table: Tabular
            a pylatex tabular object
        width: str
            The width of the image
        placement: str
            Placement of the figure, `None` is also accepted.
        """
        self.latex_name = "figure"

        if width is not None:
            if self.escape:
                width = escape_latex(width)

            width = 'width=' + str(width)

        if placement is not None:
            self.append(placement)

        self.append(NoEscape(table.dumps()))


class TableSubFigure(SubFigure):
    latex_name = "subfigure"

    def add_table(self, table, *, width=NoEscape(r'0.8\textwidth'),
            placement=NoEscape(r'\centering')):
        """Add a table to the figure.
        Args
        ----
        table: Tabular
            a pylatex tabular object
        width: str
            The width of the image
        placement: str
            Placement of the figure, `None` is also accepted.
        """

        if width is not None:
            if self.escape:
                width = escape_latex(width)

            width = 'width=' + str(width)

        if placement is not None:
            self.append(placement)

        self.append(NoEscape(table.dumps()))


def ordinal(n):
    return "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])

def decode(d, featureNum, num):
    return d['features'][featureNum]['encoder'].inverse_transform(num)
