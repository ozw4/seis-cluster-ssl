"""Thin CLI for the isolated Parihaka MAE/large decoder-seed repeat."""

from seis_ssl_cluster.parihaka.channel_seed_repeat import build_parser, main

__all__ = ['build_parser', 'main']

if __name__ == '__main__':
	main()
