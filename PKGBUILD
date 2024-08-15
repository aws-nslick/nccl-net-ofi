# Maintainer: Nicholas Sielicki <nslick@amazon.com>
pkgname=libnccl-net-ofi-aws
pkgver() { cat "libnccl-net-ofi/.version" }
pkgrel=0
pkgdesc='This is a plugin which lets EC2 developers use libfabric as network provider while running NCCL applications.'
arch=('amd64')
depends=('libfabric1-aws' 'hwloc')
makedepends=('libfabric-aws-dev' 'libhwloc-dev' 'cuda-cudart-dev-12-5' 'autoconf' 'automake' 'build-essential')
provides=('libnccl-net-ofi')
conflicts=('libnccom-net-ofi' 'libnccom-net-ofi-aws')
backup=()
options=()
license=('Apache-2.0')
url='https://github.com/aws/aws-ofi-nccl'

source=('https://github.com/aws/aws-ofi-nccl/releases/download/v1.10.0-aws/aws-ofi-nccl-1.10.0-aws.tar.gz')
sha256sums=('5f46b94003e7190c72f13711431c8bfdf3e2a76245846d2fbda9f1570f27daae')


prepare() {
    cd "libnccl-net-ofi"
    autoreconf -ivf
    ./configure --prefix=/opt/amazon/libnccl-net-ofi \
        --enable-platform-aws \
        --disable-tests \
        --with-cuda=/usr/local/cuda-12 \
        --with-libfabric=/opt/amazon/efa
}

build() {
    cd "libnccl-net-ofi"
    make -j
}

package() {
    cd "libnccl-net-ofi"
    make DESTDIR="${pkgdir}/" install
}

# vim: set sw=4 expandtab:
