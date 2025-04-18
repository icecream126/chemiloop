# ChemiLoop: Large language model for chromophore

For wavelength, f_osc, and energy calculation, you should run this code for environment setting
```
cd /home/khm/chromophore/configs/
tar xvf xtb-6.7.1-linux-x86_64.tar.xz
export PATH=$PATH:/home/khm/chromophore/configs/xtb-dist/bin
rm -rf xtb4stda/
git clone https://github.com/grimme-lab/xtb4stda.git
mkdir xtb4stda/exe
cd xtb4stda/exe
wget https://github.com/grimme-lab/stda/releases/download/v1.6.3/xtb4stda
wget https://github.com/grimme-lab/stda/releases/download/v1.6.3/stda_v1.6.3
chmod +x *
export XTB4STDAHOME=/home/khm/chromophore/configs/xtb4stda
export PATH=$PATH:$XTB4STDAHOME/exe
```
>>>>>>> initial commit
