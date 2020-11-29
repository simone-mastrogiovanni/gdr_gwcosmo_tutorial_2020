from ligo.gracedb.rest import GraceDb
import argparse

def download_skymap(id, db, args):
	outfile = open("{0}_skymap.fits.gz".format(id), "wb")
	try:
		# Use LALInference skymap if possible
		r = db.files(id, "LALInference.fits.gz")
		if args.verbose:
			print("Using LALInference skymap for {0}".format(id))
	except:
		r = db.files(id, "bayestar.fits.gz")
		if args.verbose:
			print("Using Bayestar skymap for {0}".format(id))

	outfile.write(r.read())
	outfile.close()

def main():
	parser = argparse.ArgumentParser(description = "Download skymaps from a list of events")
	parser.add_argument("event", nargs="+", help = "A list of gravitational-wave events, can be either GID for GW event or SID for superevent")
	parser.add_argument("--verbose", action = "store_true", help = "Be very verbose")
	args = parser.parse_args()
	# FIXME Make sure that you have a valid proxy
	client = GraceDb()

	for event_id in args.event:
		download_skymap(event_id, client, args)
	
if __name__ == "__main__":
	main()
