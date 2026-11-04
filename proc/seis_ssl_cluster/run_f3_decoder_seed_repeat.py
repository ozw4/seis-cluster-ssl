"""Run the isolated F3 LocalBT v3 decoder seed repeat."""

from seis_ssl_cluster.f3.lithology.decoder_seed_repeat import build_parser, main

__all__ = ['build_parser', 'main']

if __name__ == '__main__':
	main()
