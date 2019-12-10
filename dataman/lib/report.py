import io
import urllib
import base64

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 15, 8


def fig2html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    string = base64.b64encode(buf.read())

    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "{}"/>'.format(uri)
    return html
