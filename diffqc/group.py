from glob import glob
import os

def createWebPage(wp):
    with open(wp['filePath'], 'w') as fp:
        fp.write("<html>\n\t<body bgcolor=#FFF text=#000 style=\"font-family: Arial, Tahoma\">\n\n")
        fp.write("\t<script language=\"JavaScript\">\n\t\tfunction toggle(divClass) {\n\t\t\tclassDivs = document.getElementsByClassName(divClass);\n")
        fp.write("\t\t\tfor (var i=0; i<classDivs.length; i++) {\n\t\t\t\tdiv = classDivs[i];\n")
        fp.write("\t\t\t\tif (div.style.display === \"none\") {\n\t\t\t\t\tdiv.style.display = \"block\";\n\t\t\t\t} else {\n\t\t\t\t\tdiv.style.display = \"none\";\n\t\t\t\t}\n")
        fp.write("\t\t\t}\n\t\t}\n\t</script>\n\n")
        fp.write("\t<div id=\"menu\" style=\"width: 250px; top: 30px; z-index: 999; position: fixed; border: 1px solid gray; border-radius: 7px; padding: 10px; background-color: #EEE;\">\n")
        fp.write("\t\t<font size=5>Quality</font><p>\n")
        fp.write("\t\t<form>\n")
        for item in wp['maxList']:
            fp.write("\t\t\t<input type=\"checkbox\" onclick=\"toggle('" + str(item) + "')\" checked>" + str(item).replace('_',' ') + "</input><br>\n")
        fp.write("\t\t</form>\n<p>\n\n")
        fp.write("\t\t<div style=\"margin-left: 10px; float: right; padding: 3px; background-color: #CCC; border: 1px solid gray; border-radius: 7px;\" onclick=\"javascript: document.body.scrollTop = 0; document.documentElement.scrollTop = 0;\"><font size=2>scroll to top</font></div>\n")
        fp.write("\t</div>\n\n\t<div id=\"content\" style=\"margin-left: 280px; position: absolute; top: 10px; padding: 3px; border: 1px solid gray; border-radius: 7px; background-color: #FFF;\">\n")
        fp.write("\t\t<table>\n")

        # loop over subjects
        for subject_label in wp['subjects']:
            fp.write("\t\t\t<tr><td colspan=" + str(wp['maxImg']) + " bgcolor=#EEE><center><font size=3><b>sub-" + subject_label + "</b></font></center></td></tr>\n")
            # loop over images
            figures = glob(os.path.join(wp['figFolder'], "sub-%s"%subject_label,"*.png" ))
            for image_file in sorted(figures):
                # calcualte average mask size in voxels
                fp.write("\t\t\t\t<td><div name=\"" + subject_label + "\" class=\"" + os.path.basename(image_file)[0:-4] + "\"><image src=\"" + image_file.replace(os.path.dirname(wp['filePath']) + os.sep, "") + "\" width=\"100%\"></div></td>\n")

        fp.write("\t\t</table>\n\t</div>\t</body>\n</html>")
