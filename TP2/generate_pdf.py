
import os
import re
import sys
from fpdf import FPDF
import matplotlib.pyplot as plt

# Ensure matplotlib backend is non-interactive
plt.switch_backend('Agg')

class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('DejaVu', 'I', 8)
            self.cell(0, 10, 'Rapport TP2 - Classification d\'Images', new_x="Right", new_y="Top", align='R')
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', new_x="Right", new_y="Top", align='C')

def clean_latex(latex):
    # Fix math symbols not supported by mathtext
    latex = latex.replace(r'\lVert', r'\|').replace(r'\rVert', r'\|')
    return latex

def render_math(latex, filename):
    cleaned = latex.replace('$$', '').strip()
    cleaned = clean_latex(cleaned)
    if not cleaned: return False

    fig = plt.figure(figsize=(6, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    try:
        ax.text(0.5, 0.5, f"${cleaned}$", size=14, ha='center', va='center')
        fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Math Error: {e}")
        plt.close(fig)
        return False

def generate_pdf(md_path, pdf_out):
    pdf = PDF()
    
    # Add fonts
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", style="", fname=font_path)
        pdf.add_font("DejaVu", style="B", fname="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        pdf.add_font("DejaVu", style="I", fname="/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf")
        main_font = "DejaVu"
    else:
        print("Warning: DejaVu font not found. Using Helvetica.")
        main_font = "Helvetica"
        
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font(main_font, size=11)
    
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
            
        # Headers
        if line.startswith('#'):
            level = len(line.split()[0])
            txt = line[level:].strip()
            pdf.ln(5)
            pdf.set_font(main_font, 'B', 16 - (level-1)*2)
            pdf.multi_cell(0, 10, txt)
            pdf.set_font(main_font, size=11)
            continue
            
        # Images: ![alt](path)
        img_match = re.search(r'!\[.*?\]\((.*?)\)', line)
        if img_match:
            rel_path = img_match.group(1)
            full_path = rel_path
            if not os.path.exists(full_path):
                 full_path = os.path.join(os.path.dirname(md_path), rel_path)
            
            if os.path.exists(full_path):
                try:
                    pdf.ln(2)
                    # Get image dims to scale properly? FPDF handles it if we set w
                    pdf.image(full_path, w=150, x=(210-150)/2)
                    pdf.ln(2)
                except Exception as e:
                    pdf.set_text_color(255, 0, 0)
                    pdf.cell(0, 10, f"[Error: {rel_path}]", 0, 1)
                    pdf.set_text_color(0, 0, 0)
            else:
                 # Try adding TP2 prefix if running from root
                 alt = os.path.join("TP2", rel_path)
                 if os.path.exists(alt):
                     try:
                        pdf.ln(2)
                        pdf.image(alt, w=150, x=(210-150)/2)
                        pdf.ln(2)
                     except: pass
                 else:
                    pdf.set_text_color(255, 0, 0)
                    pdf.cell(0, 10, f"[Image Missing: {rel_path}]", 0, 1)
                    pdf.set_text_color(0, 0, 0)
            continue

        # Math: $$...$$
        if line.startswith('$$') and line.endswith('$$'):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fname = tmp.name
            
            if render_math(line, fname):
                pdf.ln(2)
                try:
                     pdf.image(fname, h=10, x=30) # Height 10mm
                except:
                     pass
                os.remove(fname)
            else:
                pdf.multi_cell(0, 5, line)
            continue

        # Text
        parts = re.split(r'(\*\*.*?\*\*)', line)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                pdf.set_font(main_font, 'B', 11)
                try:
                    pdf.write(5, part[2:-2])
                except:
                    pdf.write(5, part[2:-2].encode('latin-1', 'replace').decode('latin-1'))
            else:
                pdf.set_font(main_font, '', 11)
                try:
                    pdf.write(5, part)
                except:
                    # Fallback for encoding if DejaVu fails or text has weird chars
                     pdf.write(5, part.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(6)

    pdf.output(pdf_out)
    print(f"Generated {pdf_out}")

if __name__ == '__main__':
    generate_pdf('TP2/RELATORIO_TP2_FR.md', 'TP2/RELATORIO_TP2_FR.pdf')
