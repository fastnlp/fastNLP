import logging
import sys
from logging import getLevelName

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

__all__ = []

if tqdm is not None:
    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.INFO):
            super().__init__(level)

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)
else:
    class TqdmLoggingHandler(logging.StreamHandler):
        def __init__(self, level=logging.INFO):
            super().__init__(sys.stdout)
            self.setLevel(level)


class StdoutStreamHandler(logging.StreamHandler):
    """
    重载 StreamHandler 使得替换 sys.stdout 的时候能够生效。

    """
    def __init__(self):
        super(StdoutStreamHandler, self).__init__()

    def flush(self):
        """
        Flushes the stream.
        """
        self.acquire()
        try:
            sys.stdout.flush()
        finally:
            self.release()

    def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)
            stream = sys.stdout
            # issue 35046: merged two stream.writes into one.
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)

    def setStream(self, stream):
        """
        Sets the StreamHandler's stream to the specified value,
        if it is different.

        Returns the old stream, if the stream was changed, or None
        if it wasn't.
        """
        raise RuntimeError("Cannot set the stream of FStreamHandler.")

    def __repr__(self):
        level = getLevelName(self.level)
        name = getattr(sys.stdout, 'name', '')
        #  bpo-36015: name can be an int
        name = str(name)
        if name:
            name += ' '
        return '<%s %s(%s)>' % (self.__class__.__name__, name, level)
