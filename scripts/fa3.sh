mkdir -p fa3
cd fa3

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
uv run python setup.py install

cd /
rm -rf fa3