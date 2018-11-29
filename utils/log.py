from logging import getLogger, Formatter, StreamHandler, INFO
import warnings


# Logger
warnings.simplefilter("ignore", UserWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))

logger = getLogger('spacenet')
logger.setLevel(INFO)
logger.addHandler(handler)