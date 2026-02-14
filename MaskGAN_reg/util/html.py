import os

try:
    import dominate
    from dominate.tags import meta, h3, table, tr, td, p, a, img, br
    _HAVE_DOMINATE = True
except ImportError:
    # 在无 dominate 的环境下，仅占位，不做任何 HTML 可视化
    _HAVE_DOMINATE = False


class HTML:
    """HTML 可视化包装类。

    说明：
        - 如果环境中安装了 dominate，则按原逻辑生成 HTML.
        - 如果没有 dominate（如服务器无网络无法安装），则所有方法变为 no-op，
          不影响训练，只是不生成网页。
    """

    def __init__(self, web_dir, title, refresh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        if _HAVE_DOMINATE:
            self.doc = dominate.document(title=title)
            if refresh > 0:
                with self.doc.head:
                    meta(http_equiv="refresh", content=str(refresh))
        else:
            self.doc = None

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file"""
        if not _HAVE_DOMINATE or self.doc is None:
            return
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file"""
        if not _HAVE_DOMINATE or self.doc is None:
            return
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        """save the current content to the HMTL file"""
        if not _HAVE_DOMINATE or self.doc is None:
            return
        html_file = '%s/index.html' % self.web_dir
        with open(html_file, 'wt') as f:
            f.write(self.doc.render())
