import os


class HtmlVisualizer():

    def __init__(self, title="", styles="", body=""):
        self.title = title
        self.styles = styles
        self.body = body
        self.tr = ""

    def add_styles(self, style):
        self.styles += style

    def add_body(self, body):
        self.body += body

    def add_hr(self):
        self.body += "<hr>"

    def add_h1(self, h1):
        self.body += "<h1>{}</h1>".format(h1)

    def add_h2(self, h2):
        self.body += "<h1>{}</h1>".format(h2)

    def add_table(self, table):
        self.body += "<table>{}</table>".format(table)

    def add_p(self, p):
        self.tr += "<td><p>{}</p></td>".format(p)

    def add_tr(self):
        self.body += "<tr>{}</tr>".format(self.tr)
        self.tr = ""

    def add_pic(self, img_src):
        self.body += "<img src='{}'></img>".format(img_src)

    def add_img_src(self, img_src):
        self.tr += "<td><img src='{}'></img></td>".format(img_src)

    def output_html_to(self, path):
        if not path.endswith('.html'):
            path += '.html'

        html = """<!DOCTYPE html><html><head><title> """ \
               + self.title + """</title><style type="text/css">""" \
               + self.styles + """</style></head><body><table>""" \
               + self.body + """</table></body></html>"""

        with open(path, "w") as f:
            f.write(html)


def write_html(dst_path):
    visualizer = HtmlVisualizer()
    for root, subdirs, files in os.walk(dst_path):
        if files:
            root = os.path.basename(root)
            visualizer.add_h1('========================')
            visualizer.add_h1(root)
            for f in files[:100]:
                if f.endswith('png'):
                    img = os.path.join(root, f)
                    visualizer.add_pic(img)
    visualizer.output_html_to(os.path.join(dst_path, 'vis.html'))


def fn(data_path, dst_path):
    visualizer = HtmlVisualizer()

    img_list = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    for img in img_list:
        visualizer.add_pic(img)

    dst_name = os.path.basename(data_path)
    visualizer.output_html_to(os.path.join(dst_path, f'{dst_name}.html'))

