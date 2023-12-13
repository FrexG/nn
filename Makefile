CC := gcc
CFLAGS := -Wall -Wextra -g

MAIN := nn
OBJS := nn.o
SRCS := nn.c
LIBS := -lm

${MAIN}:${OBJS}
	${CC} ${CFLAGS} -o ${MAIN} ${OBJS} ${LIBS}

${OBJS}:${SRCS}
	${CC} ${CFLAGS} -c ${SRCS} -o ${OBJS} ${LIBS}

clean:
	rm -f ${OBJS} ${MAIN}
